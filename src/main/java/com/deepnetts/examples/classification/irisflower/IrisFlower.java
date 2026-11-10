package com.deepnetts.examples.classification.irisflower;

import com.deepnetts.examples.classification.irisflower.dto.Flower;
import com.deepnetts.examples.classification.irisflower.dto.SpeciesPrediction;
import com.deepnetts.examples.classification.irisflower.enums.Species;
import com.deepnetts.examples.classification.irisflower.pipeline.FlowerEncoder;
import com.deepnetts.examples.classification.irisflower.pipeline.FlowerValidator;
import com.deepnetts.examples.classification.irisflower.pipeline.SpeciesConfusion;
import com.deepnetts.examples.classification.irisflower.pipeline.SpeciesPredictor;

import deepnetts.core.DeepNetts;
import deepnetts.data.DataSets;
import deepnetts.data.MLDataItem;
import deepnetts.data.TabularDataSet;
import deepnetts.data.TrainTestSplit;
import deepnetts.data.norm.MinMaxScaler;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.NeuralNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.net.train.opt.OptimizerType;
import deepnetts.util.RandomGenerator;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.visrec.ml.data.DataSet;

// Iris flower example: a neural network learns from measured iris flowers.
// Then it tells you which of the three species a new flower is: setosa, versicolor or virginica.
public class IrisFlower {

    private static final String DATASET = "datasets/iris-flowers.csv";
    private static final String MODEL   = "models/iris-flower.dnet";
    private static final int    SEED    = 42;

    // Hides the detailed training log.
    private static final Logger DEEPNETTS_LOG = Logger.getLogger("deepnetts");

    public static void main(String[] args) {
        // Prints a clear error (missing file, bad CSV row, invalid YourFlower) and stops.
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

        // Validates YourFlower before training starts.
        Flower flower = YourFlower.create();
        FlowerValidator.validate(flower, "YourFlower");

        FlowerEncoder encoder = new FlowerEncoder();

        // ── 1. Data ──────────────────────────────
        // Loads the CSV: each row becomes a flower + its species (3 numbers, a 1 at the right species).
        TabularDataSet<MLDataItem> data = encoder.load(DATASET);

        // Counts the flowers of each species, for the dataset summary.
        int[] countPerSpecies = new int[FlowerEncoder.NUM_OUTPUTS];
        for (MLDataItem item : data) {
            countPerSpecies[encoder.toSpecies(item.getTargetOutput().getValues()).ordinal()]++;
        }
        Report.dataset(data.size(), countPerSpecies);

        // Splits the data: 80% for training, 20% for the final test on flowers the network never saw.
        TrainTestSplit split = data.trainTestSplit(0.8);
        DataSet<MLDataItem> trainingSet = split.getTrainingSet();
        DataSet<MLDataItem> testSet     = split.getTestSet();

        // Puts all inputs on the same 0-1 scale, so no input dominates.
        // Important: calculate the min and max from the training set only, not the test set (data leakage).
        MinMaxScaler scaler = DataSets.scaleToMinMax(trainingSet);
        scaler.apply(testSet);

        // ── 2. Training ──────────────────────────
        Report.trainingStarted();

        // Builds the network layer by layer using the builder pattern.
        FeedForwardNetwork network = FeedForwardNetwork.builder()
                // Input layer: receives the four flower measurements.
                .addInputLayer(FlowerEncoder.NUM_INPUTS)
                // Hidden layer: 8 units that look for patterns in those numbers.
                // TANH is an activation function: it squeezes each unit's output into the range -1 to 1.
                .addFullyConnectedLayer(8, ActivationType.TANH)
                // Output layer: 3 numbers, one probability per species.
                // SOFTMAX makes them all positive and makes them add up to 1, so the network has to choose.
                .addOutputLayer(FlowerEncoder.NUM_OUTPUTS, ActivationType.SOFTMAX)
                // Cross entropy measures how wrong the predictions are.
                .lossFunction(LossType.CROSS_ENTROPY)
                .build();

        network.getTrainer()
                // An epoch is one pass over the data.
                .setStopEpochs(2000)
                // The learning rate is the step size.
                // Too small = learning is very slow.
                // Too large = the network jumps past the best solution.
                .setLearningRate(0.01f)
                // Adam: an algorithm that improves the network step by step, adapting the step size as it goes.
                .setOptimizer(OptimizerType.ADAM)
                // Shuffles the examples each epoch, so the network doesn't learn their order.
                .setShuffle(true)
                // Early stopping: ends training when the results on the held-out data stop improving.
                // Deep Netts then keeps the network from the last epoch, not the best one.
                .setEarlyStopping(true)
                // Checks the results every 5 epochs.
                .setCheckpointEpochs(5)
                // Stops after 3 checks in a row without improvement.
                .setEarlyStoppingPatience(3);

        // Trains the network.
        // 20% of the training data is set aside (held-out data): it is used only to check progress, not to learn.
        // The test set stays untouched until the final evaluation.
        network.getTrainer().train(trainingSet, 0.2);
        Report.trainingFinished(network.getTrainer().getCurrentEpoch());

        // ── 3. Evaluation ────────────────────────
        // Checks how well the network works on the 20% of flowers it never saw:
        // for each one, compares the network's answer (the species with the highest probability) with the real species.
        List<MLDataItem> testItems = testSet.stream().toList();
        Species[] actual    = new Species[testItems.size()];
        Species[] predicted = new Species[testItems.size()];
        for (int i = 0; i < testItems.size(); i++) {
            MLDataItem item = testItems.get(i);
            network.setInput(item.getInput());
            predicted[i] = encoder.toSpecies(network.getOutput());
            actual[i]    = encoder.toSpecies(item.getTargetOutput().getValues());
        }
        Report.evaluation(SpeciesConfusion.of(actual, predicted));

        // ── 4. Save, load and predict ────────────
        // Saves the network together with its scale to models/iris-flower.dnet.
        // It can be used later without training again.
        network.setNormalizer(scaler);
        Files.createDirectories(Path.of(MODEL).getParent());
        network.save(MODEL);

        // Loads the saved network and predicts YourFlower, without training again.
        FeedForwardNetwork loadedNetwork = NeuralNetwork.load(MODEL, FeedForwardNetwork.class);
        SpeciesPrediction prediction = new SpeciesPredictor(loadedNetwork, encoder).predict(flower);
        Report.prediction(flower, prediction);
    }
}
