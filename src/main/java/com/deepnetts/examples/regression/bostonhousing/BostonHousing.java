package com.deepnetts.examples.regression.bostonhousing;

import com.deepnetts.examples.regression.bostonhousing.dto.HomeValuePrediction;
import com.deepnetts.examples.regression.bostonhousing.dto.Neighborhood;
import com.deepnetts.examples.regression.bostonhousing.pipeline.HomeValuePredictor;
import com.deepnetts.examples.regression.bostonhousing.pipeline.NeighborhoodEncoder;
import com.deepnetts.examples.regression.bostonhousing.pipeline.NeighborhoodValidator;

import deepnetts.core.DeepNetts;
import deepnetts.data.MLDataItem;
import deepnetts.data.TabularDataSet;
import deepnetts.data.TrainTestSplit;
import deepnetts.eval.Evaluators;
import deepnetts.eval.RegressionMetrics;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.NeuralNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.net.train.opt.OptimizerType;
import deepnetts.util.RandomGenerator;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.DoubleSummaryStatistics;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.visrec.ml.data.DataSet;

// Home value example: a neural network learns from known neighborhoods.
// Then it tells you the typical home value in a new neighborhood.
public class BostonHousing {

    private static final String DATASET = "datasets/boston-housing.csv";
    private static final String MODEL   = "models/boston-housing.dnet";
    private static final int    SEED    = 42;

    // Hides the detailed training log.
    private static final Logger DEEPNETTS_LOG = Logger.getLogger("deepnetts");

    public static void main(String[] args) {
        // Prints a clear error (missing file, bad CSV row, invalid YourNeighborhood) and stops.
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

        // Validates YourNeighborhood before training starts.
        Neighborhood neighborhood = YourNeighborhood.create();
        NeighborhoodValidator.validate(neighborhood, "YourNeighborhood");
        // The network learns inputs from 0 to 1 (each value divided by the largest one in the data): warn if yours is outside.
        NeighborhoodValidator.warnIfOutsideData(neighborhood, "YourNeighborhood");

        NeighborhoodEncoder encoder = new NeighborhoodEncoder();

        // ── 1. Data ──────────────────────────────
        // Loads the CSV: each row becomes a neighborhood + a target (medianHomeValueKusd).
        // The inputs are already on a small scale (see FeatureExtractor), and so is the target (see NeighborhoodEncoder).
        TabularDataSet<MLDataItem> data = encoder.load(DATASET);

        // The smallest, largest and average value to predict, in real units, for the dataset summary.
        DoubleSummaryStatistics targets = new DoubleSummaryStatistics();
        for (MLDataItem item : data) {
            targets.accept(NeighborhoodEncoder.unscaleTarget(item.getTargetOutput().getValues()[0]));
        }
        Report.dataset(targets);

        // Splits the data: 80% for training, 20% for the final test on neighborhoods the network never saw.
        TrainTestSplit split = data.trainTestSplit(0.8);
        DataSet<MLDataItem> trainingSet = split.getTrainingSet();
        DataSet<MLDataItem> testSet     = split.getTestSet();

        // ── 2. Training ──────────────────────────
        Report.trainingStarted();

        // Builds the network layer by layer using the builder pattern.
        FeedForwardNetwork network = FeedForwardNetwork.builder()
                // Input layer: receives the neighborhood numbers.
                .addInputLayer(NeighborhoodEncoder.NUM_INPUTS)
                // Hidden layer: 16 units that look for patterns in those numbers.
                // TANH is an activation function: it squeezes each unit's output into the range -1 to 1.
                .addFullyConnectedLayer(16, ActivationType.TANH)
                // Output layer: 1 number (the typical home value).
                // LINEAR means no squeezing: the output can be any number, which is what regression needs.
                // (Classification used SIGMOID here, to keep the output between 0 and 1.)
                .addOutputLayer(NeighborhoodEncoder.NUM_OUTPUTS, ActivationType.LINEAR)
                // Mean squared error measures how far the predictions are from the real values.
                // (Classification used cross entropy here.)
                .lossFunction(LossType.MEAN_SQUARED_ERROR)
                .build();

        network.getTrainer()
                // An epoch is one pass over the data.
                .setStopEpochs(2000)
                // Deep Netts also stops by itself once the error is small (the default "stop error"). The error of a scaled
                // target is small from the first epoch, so that would end training after 1 epoch. 0 turns it off.
                .setStopError(0f)
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
                // Stops after 4 checks in a row without improvement.
                // The error of a regression model falls slowly and unevenly, so a bit more patience than in classification.
                .setEarlyStoppingPatience(4);

        // Trains the network.
        // 20% of the training data is set aside (held-out data): it is used only to check progress, not to learn.
        // The test set stays untouched until the final evaluation.
        network.getTrainer().train(trainingSet, 0.2);
        Report.trainingFinished(network.getTrainer().getCurrentEpoch());

        // ── 3. Evaluation ────────────────────────
        // Checks how well the network works on the 20% of neighborhoods it never saw.
        // Deep Netts compares the network's answer with the real value for each of them.
        // Its errors are on the network's small scale, so unscaleTarget turns them back into real units below.
        RegressionMetrics metrics = Evaluators.evaluateRegressor(network, testSet);

        // The baseline: always answering with the average of the training data. The network must beat it.
        double trainingSum = 0;
        for (MLDataItem item : trainingSet) {
            trainingSum += NeighborhoodEncoder.unscaleTarget(item.getTargetOutput().getValues()[0]);
        }
        double trainingAverage = trainingSum / trainingSet.size();

        // For every test row: the real value, the network's answer and the baseline's error, in real units.
        int testCount = testSet.size();
        double[] actual    = new double[testCount];
        double[] predicted = new double[testCount];
        double baselineErrorSum = 0;
        for (int i = 0; i < testCount; i++) {
            MLDataItem item = testSet.get(i);
            network.setInput(item.getInput());
            predicted[i] = NeighborhoodEncoder.unscaleTarget(network.getOutput()[0]);
            actual[i]    = NeighborhoodEncoder.unscaleTarget(item.getTargetOutput().getValues()[0]);
            baselineErrorSum += Math.abs(actual[i] - trainingAverage);
        }
        Report.evaluation(actual, predicted,
                NeighborhoodEncoder.unscaleTarget(metrics.getMeanAbsoluteError()),
                baselineErrorSum / testCount,
                NeighborhoodEncoder.unscaleTarget(metrics.getRootMeanSquaredError()),
                metrics.getR2());

        // ── 4. Save, load and predict ────────────
        // Saves the network to models/boston-housing.dnet.
        // It can be used later without training again.
        Files.createDirectories(Path.of(MODEL).getParent());
        network.save(MODEL);

        // Loads the saved network and predicts YourNeighborhood, without training again.
        FeedForwardNetwork loadedNetwork = NeuralNetwork.load(MODEL, FeedForwardNetwork.class);
        HomeValuePrediction prediction = new HomeValuePredictor(loadedNetwork, encoder).predict(neighborhood);
        Report.prediction(neighborhood, prediction);
    }
}
