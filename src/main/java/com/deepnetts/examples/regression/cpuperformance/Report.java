package com.deepnetts.examples.regression.cpuperformance;

import com.deepnetts.examples.regression.cpuperformance.dto.Cpu;
import com.deepnetts.examples.regression.cpuperformance.dto.PerformancePrediction;
import com.deepnetts.examples.regression.cpuperformance.pipeline.CpuEncoder;

import java.util.DoubleSummaryStatistics;
import java.util.Locale;

// Prints the output, so CpuPerformance contains only the workflow.
final class Report {

    // Every section line is this wide, for example "--- 1. Dataset -------...".
    private static final int WIDTH = 80;

    private static final int EXAMPLES_SHOWN = 5;

    private Report() {
    }

    // Prints how many CPUs there are and what the values to predict look like.
    static void dataset(DoubleSummaryStatistics target) {
        long totalCount = target.getCount();
        double min = target.getMin();
        double max = target.getMax();

        section("1. Dataset");
        System.out.printf(Locale.ROOT, "  Total:      %4d CPUs%n", totalCount);
        System.out.printf(Locale.ROOT, "  Score: from %.1f to %.1f, on average %.1f points%n", min, max, target.getAverage());
        System.out.printf(              "  Inputs:     %4d numbers per CPU (the features the network learns from)%n", CpuEncoder.NUM_INPUTS);
        System.out.println();
    }

    // Prints the results on the test set (CPUs the network never saw).
    // For every test CPU we know the real value (actual) and the network's answer (predicted).
    static void evaluation(double[] actual, double[] predicted,
                           double mae, double baselineMae, double rmse, double r2) {
        int total = actual.length;

        section("3. Evaluation");
        System.out.println();
        System.out.printf(Locale.ROOT, "  Results on %d CPUs the network was never trained on:%n", total);
        System.out.println();
        System.out.printf(              "  First %d examples (points):%n", Math.min(EXAMPLES_SHOWN, total));
        System.out.printf("  %10s   %10s   %10s%n", "Actual", "Predicted", "Error");
        for (int i = 0; i < Math.min(EXAMPLES_SHOWN, total); i++) {
            System.out.printf(Locale.ROOT, "  %10.1f   %10.1f   %+10.1f%n", actual[i], predicted[i], predicted[i] - actual[i]);
        }
        System.out.println();

        System.out.printf(Locale.ROOT, "  MAE       %.1f points  - on average the prediction is off by this much%n", mae);
        System.out.printf(Locale.ROOT, "  Baseline  %.1f points  - always answering with the average would be off by this much%n", baselineMae);
        System.out.printf(Locale.ROOT, "  RMSE      %.1f points  - like MAE, but big mistakes count more%n", rmse);
        System.out.printf(Locale.ROOT, "  R2        %.3f  - share of the variation explained (1.0 = perfect, 0.0 = no better than the average)%n", r2);
        System.out.println();
    }

    // Prints your CPU and what the network says about it.
    static void prediction(Cpu cpu, PerformancePrediction prediction) {
        section("4. Your CPU");
        System.out.printf(Locale.ROOT, "  Machine cycle:      %d ns%n", cpu.machineCycleNs());
        System.out.printf(Locale.ROOT, "  Memory:             %d - %d KB%n", cpu.minMemoryKb(), cpu.maxMemoryKb());
        System.out.printf(Locale.ROOT, "  Cache:              %d KB%n", cpu.cacheMemoryKb());
        System.out.printf(Locale.ROOT, "  Channels:           %d - %d%n", cpu.minChannels(), cpu.maxChannels());
        System.out.println();
        System.out.printf(Locale.ROOT, "  Result: The network expects a performance score of about %.1f points.%n", prediction.performance());
        System.out.println();
        System.out.println("  Edit YourCpu.java and run again to explore how the result changes.");
    }

    // Prints when the training starts.
    static void trainingStarted() {
        section("2. Training");
        System.out.println("  Training the network...");
    }

    // Prints how many epochs the training took.
    static void trainingFinished(int epochs) {
        System.out.printf("  Stopped after %d epochs%n", epochs);
        System.out.println();
    }

    // Prints a section title: "--- 1. Dataset " followed by dashes up to the full width.
    private static void section(String title) {
        String start = "--- " + title + " ";
        System.out.println(start + "-".repeat(Math.max(0, WIDTH - start.length())));
    }
}
