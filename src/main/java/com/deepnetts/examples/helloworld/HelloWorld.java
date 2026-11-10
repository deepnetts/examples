package com.deepnetts.examples.helloworld;

import deepnetts.core.DeepNetts;
import deepnetts.data.DataSetItem;
import deepnetts.data.MLDataItem;
import deepnetts.data.TabularDataSet;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.net.train.opt.OptimizerType;
import deepnetts.tensor.Tensor1D;
import deepnetts.util.RandomGenerator;

import java.util.Locale;
import java.util.logging.Level;
import java.util.logging.Logger;

// Hello World: checks that Java and Deep Netts work, in a few seconds. Run it first.
// The smallest possible network (one input, one output) learns to turn degrees Celsius into Fahrenheit.
// No CSV file: the examples are made right here, from the formula F = 1.8 * C + 32.
public class HelloWorld {

    private static final int SEED = 42;
    private static final int REQUIRED_JAVA = 25;

    // Every section line is this wide, for example "--- 1. Setup -------...".
    private static final int WIDTH = 80;

    // The network works best with small numbers, so every temperature is divided by 100 (37 C -> 0.37).
    private static final float SCALE = 100f;

    // The temperature the network converts at the end.
    private static final float YOUR_CELSIUS = 37f;

    // Hides the detailed training log.
    private static final Logger DEEPNETTS_LOG = Logger.getLogger("deepnetts");

    public static void main(String[] args) {
        try {
            run();
        } catch (IllegalStateException e) {
            // Prints a clear error (for example a Java version that is too old) and stops.
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
        }
    }

    static void run() {

        // ── 1. Setup ─────────────────────────────
        section("1. Setup");
        int javaVersion = Runtime.version().feature();
        System.out.printf("  Java version: %d%n", javaVersion);
        if (javaVersion < REQUIRED_JAVA) {
            throw new IllegalStateException("Java " + REQUIRED_JAVA + " or newer is required, this is Java " + javaVersion);
        }
        System.out.println();

        // Fixed seed and a single thread, so you get the same results on every run.
        RandomGenerator.getDefault().initSeed(SEED);
        DeepNetts.getInstance().setMaxThreads(1);
        // Comment out this line to see the detailed training log.
        DEEPNETTS_LOG.setLevel(Level.WARNING);

        // ── 2. Data and training ─────────────────
        // 15 examples, from -40 C to 100 C: the input is Celsius, the correct answer is Fahrenheit.
        TabularDataSet<MLDataItem> data = new TabularDataSet<>(1, 1);
        data.setColumnNames(new String[] {"celsius", "fahrenheit"});
        for (int celsius = -40; celsius <= 100; celsius += 10) {
            float fahrenheit = 1.8f * celsius + 32;
            data.add(new DataSetItem(Tensor1D.of(new float[] {celsius / SCALE}), Tensor1D.of(new float[] {fahrenheit / SCALE})));
        }

        section("2. Training");
        System.out.println("  Training the network...");

        // One input (Celsius) goes straight to one output (Fahrenheit). LINEAR means the output is not squeezed:
        // the network computes weight * input + bias, and training has to find weight = 1.8 and bias = 32.
        FeedForwardNetwork network = FeedForwardNetwork.builder()
                .addInputLayer(1)
                .addOutputLayer(1, ActivationType.LINEAR)
                // Mean squared error: how far the predicted numbers are from the correct ones, on average.
                .lossFunction(LossType.MEAN_SQUARED_ERROR)
                .build();

        network.getTrainer()
                // An epoch is one pass over the 15 examples.
                .setStopEpochs(1000)
                // Training also stops when the error gets this small: the answers are then right to about 0.1 F.
                .setStopError(0.000001f)
                // The learning rate is the step size.
                .setLearningRate(0.1f)
                // SGD: after every example it moves the weight and the bias a little, in the direction that lowers the error.
                .setOptimizer(OptimizerType.SGD);
        network.getTrainer().train(data);
        System.out.printf("  Stopped after %d epochs%n", network.getTrainer().getCurrentEpoch());
        System.out.println();

        // ── 3. Predict ───────────────────────────
        section("3. Your temperature");
        float predicted = network.predict(Tensor1D.of(new float[] {YOUR_CELSIUS / SCALE})).getValues()[0] * SCALE;
        float exact = 1.8f * YOUR_CELSIUS + 32;
        System.out.printf(Locale.ROOT, "  %.0f C is %.1f F, the network says %.1f F%n", YOUR_CELSIUS, exact, predicted);
        System.out.println();
        System.out.println("  SUCCESS: Java and Deep Netts work. Go on with the next example in the main README.");
    }

    // Prints a section title: "--- 1. Setup " followed by dashes up to the full width.
    private static void section(String title) {
        String start = "--- " + title + " ";
        System.out.println(start + "-".repeat(Math.max(0, WIDTH - start.length())));
    }
}
