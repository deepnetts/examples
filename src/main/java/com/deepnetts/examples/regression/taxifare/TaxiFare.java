package com.deepnetts.examples.regression.taxifare;

import com.deepnetts.examples.regression.taxifare.dto.FareFormula;
import com.deepnetts.examples.regression.taxifare.dto.FarePrediction;
import com.deepnetts.examples.regression.taxifare.dto.Ride;
import com.deepnetts.examples.regression.taxifare.pipeline.FarePredictor;
import com.deepnetts.examples.regression.taxifare.pipeline.RideEncoder;
import com.deepnetts.examples.regression.taxifare.pipeline.RideValidator;

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
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.visrec.ml.data.DataSet;

// Linear regression example: a network learns what a taxi ride costs, from the distance and the duration.
// The network has no hidden layer, so what it learns is a plain formula: start price + price per km + price per minute.
// At the end, the example reads that formula out of the network.
public class TaxiFare {

    private static final String DATASET = "datasets/taxi-fares.csv";
    private static final String MODEL   = "models/taxi-fare.dnet";
    private static final int    SEED    = 42;

    // The formula the data was made with (see dataset-tools/make_taxi_fares.py), plus a little noise.
    // Real data doesn't tell you its formula: here it shows how close the network gets.
    private static final FareFormula FORMULA_OF_THE_DATA = new FareFormula(2.50, 1.20, 0.30);

    // Hides the detailed training log.
    private static final Logger DEEPNETTS_LOG = Logger.getLogger("deepnetts");

    public static void main(String[] args) {
        // Prints a clear error (missing file, bad CSV row, invalid YourRide) and stops.
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

        // Validates YourRide before training starts.
        Ride ride = YourRide.create();
        RideValidator.validate(ride, "YourRide");

        // Turns CSV rows and rides into the numbers the network learns from (see pipeline/RideEncoder).
        RideEncoder encoder = new RideEncoder();

        // ── 1. Data ──────────────────────────────
        // Loads the CSV: each row becomes a ride (the input) and its fare, the correct answer.
        TabularDataSet<MLDataItem> data = encoder.load(DATASET);
        Report.dataset(data.size());

        // Splits the data: 80% to train on, 20% to test on rides the network never saw.
        TrainTestSplit split = data.trainTestSplit(0.8);
        DataSet<MLDataItem> trainingSet = split.getTrainingSet();
        DataSet<MLDataItem> testSet     = split.getTestSet();

        // ── 2. Training ──────────────────────────
        Report.trainingStarted();

        // Builds the network layer by layer using the builder pattern.
        FeedForwardNetwork network = FeedForwardNetwork.builder()
                // Input layer: receives the two ride numbers (distance and duration).
                .addInputLayer(RideEncoder.NUM_INPUTS)
                // Output layer: 1 number, the fare. There is no hidden layer: the output is
                // bias + weight1 * distance + weight2 * duration, which is exactly linear regression.
                // LINEAR means the output is not squeezed into a range: a fare can be any amount.
                .addOutputLayer(RideEncoder.NUM_OUTPUTS, ActivationType.LINEAR)
                // Mean squared error: how far the predicted fares are from the real ones, on average.
                .lossFunction(LossType.MEAN_SQUARED_ERROR)
                .build();

        network.getTrainer()
                // At most 2000 epochs (passes over the data); early stopping usually ends much sooner.
                .setStopEpochs(2000)
                // Deep Netts also stops by itself once the error is small (the default "stop error"). The error of a
                // fare on the 0 to 1 scale is small from the first epoch, so that would end training too early. 0 turns it off.
                .setStopError(0f)
                // The learning rate is the step size.
                // Too small = learning is very slow.
                // Too large = the network jumps past the best solution.
                .setLearningRate(0.01f)
                // Adam: a popular way to adjust the network after each step; it adapts the step size by itself.
                .setOptimizer(OptimizerType.ADAM)
                // Shuffles the examples each epoch, so the network doesn't learn their order.
                .setShuffle(true)
                // Early stopping: ends training when the results on the held-out data stop improving.
                // Deep Netts then keeps the network from the last epoch, not the best one.
                .setEarlyStopping(true)
                // Checks the results every 5 epochs.
                .setCheckpointEpochs(5)
                // Stops after 4 checks in a row without improvement (so at least 20 epochs after the best one).
                .setEarlyStoppingPatience(4);

        // Trains the network.
        // 20% of the training data is set aside (held-out data): it is used only to check progress, not to learn.
        network.getTrainer().train(trainingSet, 0.2);
        Report.trainingFinished(network.getTrainer().getCurrentEpoch());

        // ── 3. Evaluation ────────────────────────
        // Checks how well the network works on the 20% of rides it never saw.
        RegressionMetrics metrics = Evaluators.evaluateRegressor(network, testSet);

        // The baseline: always answering with the average fare of the training rides. The network must beat it.
        double fareSum = 0;
        for (MLDataItem item : trainingSet) {
            fareSum += item.getTargetOutput().getValues()[0];
        }
        double averageFare = fareSum / trainingSet.size();
        double baselineErrorSum = 0;
        for (MLDataItem item : testSet) {
            baselineErrorSum += Math.abs(item.getTargetOutput().getValues()[0] - averageFare);
        }
        Report.evaluation(testSet.size(),
                encoder.toFareEur(metrics.getMeanAbsoluteError()),
                encoder.toFareEur((float) (baselineErrorSum / testSet.size())),
                metrics.getR2());

        // What the network learned: its two weights and its bias, read as a fare formula.
        Report.formula(encoder.learnedFormula(network), FORMULA_OF_THE_DATA);

        // ── 4. Save, load and predict ────────────
        // Saves the network to models/taxi-fare.dnet.
        // It can be used later without training again.
        Files.createDirectories(Path.of(MODEL).getParent());
        network.save(MODEL);

        // Loads the saved network and prices YourRide, without training again.
        FeedForwardNetwork loadedNetwork = NeuralNetwork.load(MODEL, FeedForwardNetwork.class);
        FarePrediction prediction = new FarePredictor(loadedNetwork, encoder).predict(ride);
        Report.prediction(ride, prediction);
    }
}
