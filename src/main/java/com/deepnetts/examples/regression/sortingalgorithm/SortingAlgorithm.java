
package com.deepnetts.examples.regression.sortingalgorithm;

import com.deepnetts.examples.regression.sortingalgorithm.dto.ArrayToSort;
import com.deepnetts.examples.regression.sortingalgorithm.dto.MeasuredRun;
import com.deepnetts.examples.regression.sortingalgorithm.dto.SortingAdvice;
import com.deepnetts.examples.regression.sortingalgorithm.pipeline.RunEncoder;
import com.deepnetts.examples.regression.sortingalgorithm.pipeline.RunValidator;
import com.deepnetts.examples.regression.sortingalgorithm.pipeline.SortingAdvisor;
import com.deepnetts.examples.regression.sortingalgorithm.pipeline.SortingEvaluator;

import deepnetts.core.DeepNetts;
import deepnetts.data.MLDataItem;
import deepnetts.data.TabularDataSet;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.NeuralNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.net.train.opt.OptimizerType;
import deepnetts.util.RandomGenerator;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

// Sorting algorithm example (regression): a neural network learns how long 9 sorting algorithms take,
// from times measured on real arrays. Then it tells you which algorithm will be the fastest for your array.
public class SortingAlgorithm {

    // Measured on one laptop with benchmark/SortingBenchmark. Run it to measure your own computer.
    private static final String DATASET = "datasets/sorting-algorithm.csv";
    private static final String MODEL   = "models/sorting-algorithm.dnet";
    private static final int    SEED    = 42;

    // 80% of the arrays for training, 20% for the final test. All 9 runs of an array go to the same side,
    // so the test arrays are new to the network, and on each of them we know which algorithm really was the fastest.
    private static final double TRAINING_SHARE = 0.8;

    // Hides the detailed training log.
    private static final Logger DEEPNETTS_LOG = Logger.getLogger("deepnetts");

    public static void main(String[] args) {
        // Prints a clear error (missing file, bad CSV row, invalid YourArray) and stops.
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

        // Validates YourArray before training starts.
        ArrayToSort array = YourArray.create();
        RunValidator.validate(array, "YourArray");
        // The network learns array sizes from 100 to 5000 (on a 0 to 1 scale): warn if yours is outside.
        RunValidator.warnIfOutsideData(array, "YourArray");

        RunEncoder encoder = new RunEncoder();

        // ── 1. Data ──────────────────────────────
        // Loads the CSV: each row is one algorithm sorting one array + the measured time.
        List<MeasuredRun> runs = encoder.loadRuns(DATASET);
        Report.dataset((int) runs.stream().map(MeasuredRun::arrayId).distinct().count(),
                runs.stream().mapToDouble(MeasuredRun::timeMicroseconds).summaryStatistics(),
                SortingEvaluator.fastestCounts(runs));

        // Splits by array (see TRAINING_SHARE).
        List<Integer> arrayIds = new ArrayList<>(runs.stream().map(MeasuredRun::arrayId).distinct().sorted().toList());
        Collections.shuffle(arrayIds, new Random(SEED));
        Set<Integer> trainingArrays = new HashSet<>(arrayIds.subList(0, (int) (arrayIds.size() * TRAINING_SHARE)));
        List<MeasuredRun> trainingRuns = runs.stream().filter(r -> trainingArrays.contains(r.arrayId())).toList();
        List<MeasuredRun> testRuns     = runs.stream().filter(r -> !trainingArrays.contains(r.arrayId())).toList();
        TabularDataSet<MLDataItem> trainingSet = encoder.toDataSet(trainingRuns);

        // ── 2. Training ──────────────────────────
        Report.trainingStarted();

        // Builds the network layer by layer using the builder pattern.
        FeedForwardNetwork network = FeedForwardNetwork.builder()
                // Input layer: which algorithm (9 numbers, one of them 1), the array size and how sorted it is.
                .addInputLayer(RunEncoder.NUM_INPUTS)
                // Two hidden layers: each algorithm reacts to size and order in its own way
                // (insertion sort loves a sorted array, selection sort doesn't care), so the network must combine the inputs.
                // TANH is an activation function: it squeezes each unit's output into the range -1 to 1.
                .addFullyConnectedLayer(32, ActivationType.TANH)
                .addFullyConnectedLayer(16, ActivationType.TANH)
                // Output layer: 1 number (the time, on a log scale).
                // LINEAR means no squeezing: the output can be any number, which is what regression needs.
                .addOutputLayer(RunEncoder.NUM_OUTPUTS, ActivationType.LINEAR)
                // Mean squared error measures how far the predictions are from the real values.
                .lossFunction(LossType.MEAN_SQUARED_ERROR)
                .build();

        network.getTrainer()
                // An epoch is one pass over the data.
                .setStopEpochs(2000)
                // Deep Netts also stops by itself once the error is small (the default "stop error"). The error of a scaled
                // target is small from the first epoch, so that would end training too early. 0 turns it off.
                .setStopError(0f)
                // The learning rate is the step size.
                // Too small = learning is very slow.
                // Too large = the network jumps past the best solution.
                // Small here: the times are measured, so they have noise, and bigger steps made the results
                // jump around from one SEED to the next.
                .setLearningRate(0.001f)
                // Adam: an algorithm that improves the network step by step, adapting the step size as it goes.
                .setOptimizer(OptimizerType.ADAM)
                // Shuffles the examples each epoch, so the network doesn't learn their order.
                .setShuffle(true)
                // Early stopping: ends training when the results on the held-out data stop improving.
                // Deep Netts then keeps the network from the last epoch, not the best one.
                .setEarlyStopping(true)
                // Checks the results every 5 epochs.
                .setCheckpointEpochs(5)
                // Stops after 8 checks in a row without improvement.
                // More than in the other examples: the noise in the times makes the error on the held-out data go up and down,
                // and with 4 checks training sometimes stopped after 25 epochs, before it had learned enough.
                .setEarlyStoppingPatience(8);

        // Trains the network.
        // 20% of the training data is set aside (held-out data): it is used only to check progress, not to learn.
        // The test arrays stay untouched until the final evaluation.
        network.getTrainer().train(trainingSet, 0.2);
        Report.trainingFinished(network.getTrainer().getCurrentEpoch());

        // ── 3. Evaluation ────────────────────────
        // Checks how well the network works on the arrays it never saw: first the predicted times,
        // then the choice that matters, whether the algorithm it predicts as the fastest really was the fastest.
        SortingAdvisor advisor = new SortingAdvisor(network, encoder);
        double[] predicted = testRuns.stream().mapToDouble(r -> advisor.time(r.run())).toArray();
        Report.evaluation(testRuns, predicted, SortingEvaluator.timeAccuracy(testRuns, predicted));
        Report.choice(SortingEvaluator.choice(trainingRuns, testRuns, predicted));

        // ── 4. Save, load and predict ────────────
        // Saves the network to models/sorting-algorithm.dnet.
        // It can be used later without training again.
        Files.createDirectories(Path.of(MODEL).getParent());
        network.save(MODEL);

        // Loads the saved network and ranks the algorithms for YourArray, without training again.
        FeedForwardNetwork loadedNetwork = NeuralNetwork.load(MODEL, FeedForwardNetwork.class);
        SortingAdvice advice = new SortingAdvisor(loadedNetwork, encoder).predict(array);
        Report.prediction(array, advice);
    }
}
