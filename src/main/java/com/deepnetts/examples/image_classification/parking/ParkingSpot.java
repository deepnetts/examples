package com.deepnetts.examples.image_classification.parking;

import com.deepnetts.examples.image_classification.parking.dto.ParkingData;
import com.deepnetts.examples.image_classification.parking.dto.SpotPrediction;
import com.deepnetts.examples.image_classification.parking.pipeline.SpotEncoder;
import com.deepnetts.examples.image_classification.parking.pipeline.SpotFileReader;
import com.deepnetts.examples.image_classification.parking.pipeline.SpotPredictor;

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

// Parking spot example: a camera looks down at a parking lot and takes a picture every 5 minutes.
// Every picture is cut into one small image per parking spot. A convolutional neural network learns
// from these images to tell if a spot is taken (busy) or free.
public class ParkingSpot {

    private static final String DATASET = "datasets/ParkingSet";
    private static final String MODEL   = "models/parking-spot.dnet";
    private static final int    SEED    = 42;

    // The last 20% of the day (in time) is the test set.
    private static final double TEST_SHARE = 0.2;
    // The camera takes a picture every 5 minutes. Training keeps every 3rd one (one every 15 minutes),
    // because pictures 5 minutes apart are almost the same and only make training slower.
    private static final int CAPTURE_MINUTES = 5;
    private static final int TRAINING_STEP = 3;

    // The network answers with a probability. From this value up, the spot is called busy.
    private static final float BUSY_THRESHOLD = 0.5f;

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
        BufferedImage image = SpotFileReader.read(imagePath, "YourImage");

        SpotEncoder encoder = new SpotEncoder();

        // ── 1. Data ──────────────────────────────
        // Loads the pictures listed in index.txt: each picture gets a target (1 = busy, 0 = free).
        // The pixel values are already on a small scale, between 0 and 1.
        // The split is by time, not random: the network learns from the morning and early afternoon,
        // and is tested on the end of the day. That is how it will be used: on pictures taken later.
        ParkingData data = encoder.load(DATASET, TEST_SHARE, TRAINING_STEP);
        ImageSet trainingSet = data.trainingSet();
        ImageSet testSet     = data.testSet();
        Report.dataset(data, countBusy(trainingSet), countBusy(testSet),
                CAPTURE_MINUTES * TRAINING_STEP);

        // ── 2. Training ──────────────────────────
        Report.trainingStarted();

        // Builds the network layer by layer using the builder pattern.
        ConvolutionalNetwork network = ConvolutionalNetwork.builder()
                // Input layer: receives the pixels of the picture (width, height, colors).
                .addInputLayer(SpotEncoder.IMAGE_WIDTH, SpotEncoder.IMAGE_HEIGHT, SpotEncoder.IMAGE_CHANNELS)
                // Convolutional layer: 8 filters of size 3x3. Each filter slides over the picture
                // and looks for one small pattern (an edge, a color change...).
                // RELU is an activation function: it keeps positive values and turns negative ones into 0.
                .addConvolutionalLayer(8, 3, ActivationType.RELU)
                // Max pooling layer: keeps the strongest value in every 2x2 square, so the picture becomes
                // two times smaller in each direction. The network cares about what is in the picture, not exactly where.
                .addMaxPoolingLayer(2)
                // Hidden layer: 32 units that combine the patterns found by the filters.
                // TANH is an activation function: it squeezes each unit's output into the range -1 to 1.
                .addFullyConnectedLayer(32, ActivationType.TANH)
                // Output layer: 1 number, the probability that the spot is taken.
                // SIGMOID keeps the output between 0 and 1.
                .addOutputLayer(SpotEncoder.NUM_OUTPUTS, ActivationType.SIGMOID)
                // Cross entropy measures how far the predicted probabilities are from the real answers.
                .lossFunction(LossType.CROSS_ENTROPY)
                .build();

        network.getTrainer()
                // An epoch is one pass over the data.
                .setStopEpochs(200)
                // The learning rate is the step size.
                // Too small = learning is very slow.
                // Too large = the network jumps past the best solution.
                .setLearningRate(0.01f)
                // SGD: the simplest algorithm: after every picture it moves the weights a little,
                // in the direction that lowers the error.
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
        // Checks how well the network works on the pictures from the end of the day:
        // for each one, compares the network's answer (the busy probability) with the real answer.
        int testCount = testSet.size();
        String[] imageNames     = new String[testCount];
        boolean[] actual        = new boolean[testCount];
        float[] busyProbability = new float[testCount];
        for (int i = 0; i < testCount; i++) {
            ExampleImage item = testSet.get(i);
            network.setInput(item.getInput());
            busyProbability[i] = network.getOutput()[0];
            actual[i]          = item.getTargetOutput().getValues()[0] == 1f;
            imageNames[i]      = item.getFile().getName();
        }

        // Deep Netts counts the right and wrong answers (the confusion matrix), with the same threshold.
        ClassifierEvaluator evaluator = new ClassifierEvaluator();
        evaluator.setThreshold(BUSY_THRESHOLD);
        evaluator.evaluate(network, testSet);
        Report.evaluation(imageNames, actual, busyProbability, BUSY_THRESHOLD, evaluator.getConfusionMatrix());

        // ── 4. Save, load and predict ────────────
        // Saves the network to models/parking-spot.dnet.
        // It can be used later without training again.
        Files.createDirectories(Path.of(MODEL).getParent());
        network.save(MODEL);

        // Loads the saved network and judges YourImage, without training again.
        ConvolutionalNetwork loadedNetwork = NeuralNetwork.load(MODEL, ConvolutionalNetwork.class);
        SpotPrediction prediction = new SpotPredictor(loadedNetwork, encoder, BUSY_THRESHOLD).predict(image);
        Report.prediction(imagePath, prediction);
    }

    // How many images in the set show a taken spot (target 1).
    private static int countBusy(ImageSet images) {
        int count = 0;
        for (ExampleImage item : images) {
            count += item.getTargetOutput().getValues()[0] == 1f ? 1 : 0;
        }
        return count;
    }
}
