package com.deepnetts.examples.classification.lightswitch;

import com.deepnetts.examples.classification.lightswitch.dto.LightPrediction;
import com.deepnetts.examples.classification.lightswitch.dto.Switches;
import com.deepnetts.examples.classification.lightswitch.dto.TruthTableRow;
import com.deepnetts.examples.classification.lightswitch.pipeline.LightPredictor;
import com.deepnetts.examples.classification.lightswitch.pipeline.SwitchesEncoder;

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
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

// XOR example: a staircase light with a switch at the bottom and one at the top.
// The light is on when exactly one switch is up (the XOR rule). No straight line separates "on" from "off",
// so a network without a hidden layer can't learn it, and a network with a hidden layer can.
// The example trains both and compares them.
public class LightSwitches {

    private static final String DATASET = "datasets/light-switches.csv";
    private static final String MODEL   = "models/light-switches.dnet";
    private static final int    SEED    = 42;

    // Hides the detailed training log.
    private static final Logger DEEPNETTS_LOG = Logger.getLogger("deepnetts");

    public static void main(String[] args) {
        // Prints a clear error (missing file, bad CSV row) and stops.
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

        Switches yourSwitches = YourSwitches.create();

        // Turns CSV rows and switches into the numbers the network learns from (see pipeline/SwitchesEncoder).
        SwitchesEncoder encoder = new SwitchesEncoder();

        // ── 1. Data ──────────────────────────────
        // Loads the CSV: the 4 possible positions of the two switches, and whether the light is on.
        // With only 4 possible inputs, the networks learn from all of them and are checked on all of them:
        // the question here is not "does it work on new data" but "can it learn the rule at all".
        TabularDataSet<MLDataItem> data = encoder.load(DATASET);
        Report.dataset(data.size());

        // ── 2. Training ──────────────────────────
        Report.trainingStarted();

        // Network 1, without a hidden layer: the output is SIGMOID(bias + weight1 * bottom + weight2 * top).
        // That can only draw one straight line between "on" and "off" (logistic regression).
        FeedForwardNetwork withoutHiddenLayer = FeedForwardNetwork.builder()
                .addInputLayer(SwitchesEncoder.NUM_INPUTS)
                .addOutputLayer(SwitchesEncoder.NUM_OUTPUTS, ActivationType.SIGMOID)
                .lossFunction(LossType.CROSS_ENTROPY)
                .build();
        train(withoutHiddenLayer, data);
        Report.trainingFinished("Without a hidden layer", withoutHiddenLayer.getTrainer().getCurrentEpoch());

        // Network 2, with a hidden layer of 4 units: each unit can draw its own line,
        // and the output combines them, so together they can cut out the "on" corners.
        FeedForwardNetwork withHiddenLayer = FeedForwardNetwork.builder()
                .addInputLayer(SwitchesEncoder.NUM_INPUTS)
                // Hidden layer: 4 units. TANH squeezes each unit's output into the range -1 to 1.
                .addFullyConnectedLayer(4, ActivationType.TANH)
                .addOutputLayer(SwitchesEncoder.NUM_OUTPUTS, ActivationType.SIGMOID)
                .lossFunction(LossType.CROSS_ENTROPY)
                .build();
        train(withHiddenLayer, data);
        Report.trainingFinished("With a hidden layer", withHiddenLayer.getTrainer().getCurrentEpoch());

        // ── 3. Evaluation ────────────────────────
        // Asks both networks about every position of the switches and compares with the real light.
        LightPredictor predictorWithout = new LightPredictor(withoutHiddenLayer, encoder);
        LightPredictor predictorWith    = new LightPredictor(withHiddenLayer, encoder);
        List<TruthTableRow> rows = new ArrayList<>();
        for (MLDataItem item : data) {
            float[] inputs = item.getInput().getValues();
            Switches switches = new Switches(inputs[0] == 1f, inputs[1] == 1f);
            rows.add(new TruthTableRow(
                    switches.bottomUp(),
                    switches.topUp(),
                    item.getTargetOutput().getValues()[0] == 1f,
                    predictorWithout.predict(switches).onProbability(),
                    predictorWith.predict(switches).onProbability()));
        }
        Report.evaluation(rows);

        // ── 4. Save, load and predict ────────────
        // Saves the network with the hidden layer to models/light-switches.dnet.
        // It can be used later without training again.
        Files.createDirectories(Path.of(MODEL).getParent());
        withHiddenLayer.save(MODEL);

        // Loads the saved network and judges YourSwitches, without training again.
        FeedForwardNetwork loadedNetwork = NeuralNetwork.load(MODEL, FeedForwardNetwork.class);
        LightPrediction prediction = new LightPredictor(loadedNetwork, encoder).predict(yourSwitches);
        Report.prediction(yourSwitches, prediction);
    }

    // The same training settings for both networks, so the only difference is the hidden layer.
    private static void train(FeedForwardNetwork network, TabularDataSet<MLDataItem> data) {
        network.getTrainer()
                // At most 5000 epochs (passes over the 4 rows). That takes well under a second.
                .setStopEpochs(5000)
                // Stops earlier once the error is this small: the network is then sure of its answers.
                .setStopError(0.01f)
                // By default Deep Netts also stops as soon as every answer is on the right side of 50%,
                // even if only just (51%). A value above 1 (100%) turns that off, so the network keeps learning
                // until it is sure.
                .setStopAccuracy(1.01f)
                // The learning rate is the step size. With 4 rows, a big step is fine.
                .setLearningRate(0.1f)
                // Adam: a popular way to adjust the network after each step; it adapts the step size by itself.
                .setOptimizer(OptimizerType.ADAM);
        // No held-out data and no early stopping: with 4 rows, the network learns from all of them.
        network.getTrainer().train(data);
    }
}
