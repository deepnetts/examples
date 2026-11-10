package com.deepnetts.examples.image_classification.digits;

import com.deepnetts.examples.image_classification.digits.dto.DigitPrediction;
import com.deepnetts.examples.image_classification.digits.pipeline.DigitConfusion;
import com.deepnetts.examples.image_classification.digits.pipeline.DigitEncoder;
import com.deepnetts.examples.image_classification.digits.pipeline.DigitFileReader;
import com.deepnetts.examples.image_classification.digits.pipeline.DigitPredictor;

import deepnetts.core.DeepNetts;
import deepnetts.data.ExampleImage;
import deepnetts.data.ImageSet;
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

// Digits example: a convolutional neural network learns to read handwritten digits, 0 to 9.
// Unlike the other image examples, the answer is not yes or no: the network picks one of 10 digits.
public class Digits {

    private static final String DATASET = "datasets/DigitSet";
    private static final String MODEL   = "models/digits.dnet";
    private static final int    SEED    = 42;

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
        BufferedImage image = DigitFileReader.read(imagePath, "YourImage");

        DigitEncoder encoder = new DigitEncoder();

        // ── 1. Data ──────────────────────────────
        // Loads the images listed in index.txt: each image gets a target, 10 numbers with a 1 at its digit.
        // The pixel values are already on a small scale, between 0 and 1.
        ImageSet data = encoder.load(DATASET);

        // Counts the images of each digit, for the dataset summary.
        int[] countPerDigit = new int[DigitEncoder.NUM_OUTPUTS];
        for (ExampleImage item : data) {
            countPerDigit[encoder.toDigit(item.getTargetOutput().getValues())]++;
        }
        Report.dataset(data.size(), countPerDigit);

        // Splits the data: 80% for training, 20% for the final test on images the network never saw.
        ImageSet[] split = data.split(0.8, 0.2);
        ImageSet trainingSet = split[0];
        ImageSet testSet     = split[1];

        // ── 2. Training ──────────────────────────
        Report.trainingStarted();

        // Builds the network layer by layer using the builder pattern.
        ConvolutionalNetwork network = ConvolutionalNetwork.builder()
                // Input layer: receives the pixels of the image (width, height, 1 gray value per pixel).
                .addInputLayer(DigitEncoder.IMAGE_WIDTH, DigitEncoder.IMAGE_HEIGHT, DigitEncoder.IMAGE_CHANNELS)
                // Convolutional layer: 8 filters of size 5x5. Each filter slides over the image
                // and looks for one small pattern (a stroke, a curve, a corner...).
                // RELU is an activation function: it keeps positive values and turns negative ones into 0.
                .addConvolutionalLayer(8, 5, ActivationType.RELU)
                // Max pooling layer: keeps the strongest value in every 2x2 square, so the image becomes
                // two times smaller in each direction. A digit is still a 7 if it moves a pixel or two.
                .addMaxPoolingLayer(2)
                // Hidden layer: 32 units that combine the patterns found by the filters.
                // TANH is an activation function: it squeezes each unit's output into the range -1 to 1.
                .addFullyConnectedLayer(32, ActivationType.TANH)
                // Output layer: 10 numbers, one probability per digit.
                // SOFTMAX makes them all positive and makes them add up to 1, so the network has to choose.
                .addOutputLayer(DigitEncoder.NUM_OUTPUTS, ActivationType.SOFTMAX)
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
                // SGD: the simplest algorithm: after every image it moves the weights a little,
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
        // Checks how well the network works on the 20% of images it never saw:
        // for each one, compares the network's answer (the digit with the highest probability) with the real digit.
        int testCount = testSet.size();
        String[] imageNames = new String[testCount];
        int[] actual        = new int[testCount];
        int[] predicted     = new int[testCount];
        float[] probability = new float[testCount];
        for (int i = 0; i < testCount; i++) {
            ExampleImage item = testSet.get(i);
            network.setInput(item.getInput());
            float[] outputs = network.getOutput();
            predicted[i]   = encoder.toDigit(outputs);
            probability[i] = outputs[predicted[i]];
            actual[i]      = encoder.toDigit(item.getTargetOutput().getValues());
            imageNames[i]  = item.getFile().getName();
        }
        Report.evaluation(imageNames, actual, predicted, probability, DigitConfusion.of(actual, predicted));

        // ── 4. Save, load and predict ────────────
        // Saves the network to models/digits.dnet.
        // It can be used later without training again.
        Files.createDirectories(Path.of(MODEL).getParent());
        network.save(MODEL);

        // Loads the saved network and reads YourImage, without training again.
        ConvolutionalNetwork loadedNetwork = NeuralNetwork.load(MODEL, ConvolutionalNetwork.class);
        DigitPrediction prediction = new DigitPredictor(loadedNetwork, encoder).predict(image);
        Report.prediction(imagePath, prediction);
    }
}
