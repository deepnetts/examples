package com.deepnetts.examples.image_classification.dukedetection;

import com.deepnetts.examples.image_classification.dukedetection.dto.DukePrediction;
import com.deepnetts.examples.image_classification.dukedetection.pipeline.DukeEncoder;
import com.deepnetts.examples.image_classification.dukedetection.pipeline.DukeFileReader;
import com.deepnetts.examples.image_classification.dukedetection.pipeline.DukePredictor;

import deepnetts.core.DeepNetts;
import deepnetts.data.ExampleImage;
import deepnetts.data.ImageSet;
import deepnetts.eval.ClassifierEvaluator;
import deepnetts.net.ConvolutionalNetwork;
import deepnetts.net.NeuralNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.net.train.opt.OptimizerType;
import deepnetts.util.RandomGenerator;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.logging.Level;
import java.util.logging.Logger;

// Duke detection example: a convolutional neural network learns from images of Duke (the Java mascot)
// and of other things. Then it tells you if a new image shows Duke.
public class DukeDetection {

    private static final String DATASET = "datasets/DukeSet";
    private static final String MODEL   = "models/duke-detection.dnet";
    private static final int    SEED    = 42;

    // The network answers with a probability. From this value up, the image is called Duke.
    private static final float DUKE_THRESHOLD = 0.5f;

    // Hides the detailed training log.
    private static final Logger DEEPNETTS_LOG = Logger.getLogger("deepnetts");

    public static void main(String[] args) {
        // Prints a clear error (missing file, bad image, invalid YourImage) and stops.
        try {
            run();
        } catch (IOException | IllegalArgumentException e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
        } catch (ClassNotFoundException e) {
            // The saved model names a class that is not on the classpath (for example another Deep Netts version).
            System.err.println("Error: cannot load the saved model, class not found: " + e.getMessage());
            System.exit(1);
        }
    }

    static void run() throws IOException, ClassNotFoundException {

        // Fixed seed and a single thread, so you get the same results on every run.
        RandomGenerator.getDefault().initSeed(SEED);
        DeepNetts.getInstance().setMaxThreads(1);
        // Comment out this line to see the detailed training log.
        DEEPNETTS_LOG.setLevel(Level.WARNING);

        // Reads YourImage before training starts.
        Path imagePath = YourImage.path();
        BufferedImage image = DukeFileReader.read(imagePath, "YourImage");

        DukeEncoder encoder = new DukeEncoder();

        // ── 1. Data ──────────────────────────────
        // Loads the images listed in index.txt: each image gets a target (1 = Duke, 0 = not Duke).
        // The pixel values are already on a small scale, between 0 and 1 (see DukeEncoder).
        ImageSet data = encoder.load(DATASET);

        // Counts the images that show Duke (target 1), for the dataset summary.
        int dukeCount = 0;
        for (ExampleImage item : data) {
            dukeCount += item.getTargetOutput().getValues()[0] == 1f ? 1 : 0;
        }
        Report.dataset(data.size(), dukeCount);

        // Splits the data: 80% for training, 20% for the final test on images the network never saw.
        ImageSet[] split = data.split(0.8, 0.2);
        ImageSet trainingSet = split[0];
        ImageSet testSet     = split[1];

        // ── 2. Training ──────────────────────────
        Report.trainingStarted();

        // Builds the network layer by layer using the builder pattern.
        ConvolutionalNetwork network = ConvolutionalNetwork.builder()
                // Input layer: receives the pixels of the image (width, height, colors).
                .addInputLayer(DukeEncoder.IMAGE_WIDTH, DukeEncoder.IMAGE_HEIGHT, DukeEncoder.IMAGE_CHANNELS)
                // Convolutional layer: 8 filters of size 3x3. Each filter slides over the image
                // and looks for one small pattern (an edge, a color change...).
                // RELU is an activation function: it keeps positive values and turns negative ones into 0.
                .addConvolutionalLayer(8, 3, ActivationType.RELU)
                // Max pooling layer: keeps the strongest value in every 2x2 square, so the image becomes
                // two times smaller in each direction. The network cares about what is in the image, not exactly where.
                .addMaxPoolingLayer(2)
                // Hidden layer: 32 units that combine the patterns found by the filters.
                // TANH is an activation function: it squeezes each unit's output into the range -1 to 1.
                .addFullyConnectedLayer(32, ActivationType.TANH)
                // Output layer: 1 number, the probability that the image shows Duke.
                // SIGMOID keeps the output between 0 and 1.
                .addOutputLayer(DukeEncoder.NUM_OUTPUTS, ActivationType.SIGMOID)
                // Cross entropy measures how far the predicted probabilities are from the real answers.
                // (Regression used mean squared error here.)
                .lossFunction(LossType.CROSS_ENTROPY)
                .build();

        network.getTrainer()
                // An epoch is one pass over the data.
                // (Training also stops earlier, as soon as the network gets all training images right.)
                .setStopEpochs(200)
                // The learning rate is the step size.
                // Too small = learning is very slow.
                // Too large = the network jumps past the best solution.
                .setLearningRate(0.01f)
                // SGD: the simplest algorithm: after every image it moves the weights a little,
                // in the direction that lowers the error.
                // (Adam, used in regression, was much slower here and the network often got stuck.)
                .setOptimizer(OptimizerType.SGD)
                // Shuffles the examples each epoch, so the network doesn't learn their order.
                .setShuffle(true)
                // Early stopping: ends training when the results on the held-out data stop improving.
                // Deep Netts then keeps the network from the last epoch, not the best one.
                .setEarlyStopping(true)
                // Checks the results every 5 epochs.
                .setCheckpointEpochs(5)
                // Stops after 4 checks in a row without improvement.
                .setEarlyStoppingPatience(4);

        // Trains the network.
        // 20% of the training data is set aside (held-out data): it is used only to check progress, not to learn.
        // The test set stays untouched until the final evaluation.
        network.getTrainer().train(trainingSet, 0.2);
        Report.trainingFinished(network.getTrainer().getCurrentEpoch());

        // ── 3. Evaluation ────────────────────────
        // Checks how well the network works on the 20% of images it never saw:
        // for each one, compares the network's answer (the Duke probability) with the real answer.
        int testCount = testSet.size();
        String[] imageNames     = new String[testCount];
        boolean[] actual        = new boolean[testCount];
        float[] dukeProbability = new float[testCount];
        for (int i = 0; i < testCount; i++) {
            ExampleImage item = testSet.get(i);
            network.setInput(item.getInput());
            dukeProbability[i] = network.getOutput()[0];
            actual[i]          = item.getTargetOutput().getValues()[0] == 1f;
            imageNames[i]      = item.getFile().getName();
        }

        // Deep Netts counts the right and wrong answers (the confusion matrix), with the same threshold.
        ClassifierEvaluator evaluator = new ClassifierEvaluator();
        evaluator.setThreshold(DUKE_THRESHOLD);
        evaluator.evaluate(network, testSet);
        Report.evaluation(imageNames, actual, dukeProbability, DUKE_THRESHOLD, evaluator.getConfusionMatrix());

        // ── 4. Save, load and predict ────────────
        // Saves the network to models/duke-detection.dnet.
        // It can be used later without training again.
        Files.createDirectories(Path.of(MODEL).getParent());
        network.save(MODEL);

        // Loads the saved network and judges YourImage, without training again.
        ConvolutionalNetwork loadedNetwork = NeuralNetwork.load(MODEL, ConvolutionalNetwork.class);
        DukePrediction prediction = new DukePredictor(loadedNetwork, encoder, DUKE_THRESHOLD).predict(image);
        Report.prediction(imagePath, prediction);
    }
}
