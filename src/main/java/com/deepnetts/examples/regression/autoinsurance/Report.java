package com.deepnetts.examples.regression.autoinsurance;

import com.deepnetts.examples.regression.autoinsurance.dto.PaymentPrediction;
import com.deepnetts.examples.regression.autoinsurance.dto.PolicyGroup;
import com.deepnetts.examples.regression.autoinsurance.pipeline.PolicyEncoder;

import java.util.DoubleSummaryStatistics;
import java.util.Locale;

// Prints the output, so AutoInsurance contains only the workflow.
final class Report {

    // Every section line is this wide, for example "--- 1. Dataset -------...".
    private static final int WIDTH = 80;

    private static final int EXAMPLES_SHOWN = 5;

    private Report() {
    }

    // Prints how many policy groups there are and what the values to predict look like.
    static void dataset(DoubleSummaryStatistics target) {
        long totalCount = target.getCount();
        double min = target.getMin();
        double max = target.getMax();

        section("1. Dataset");
        System.out.printf(Locale.ROOT, "  Total:      %4d policy groups%n", totalCount);
        System.out.printf(Locale.ROOT, "  Payment: from %.2f to %.2f, on average %.2f kSEK%n", min, max, target.getAverage());
        System.out.printf(              "  Inputs:     %4d numbers per policy group (the features the network learns from)%n", PolicyEncoder.NUM_INPUTS);
        System.out.println();
    }

    // Prints the results on the test set (policy groups the network never saw).
    // For every test policy group we know the real value (actual) and the network's answer (predicted).
    static void evaluation(double[] actual, double[] predicted,
                           double mae, double baselineMae, double rmse, double r2) {
        int total = actual.length;

        section("3. Evaluation");
        System.out.println();
        System.out.printf(Locale.ROOT, "  Results on %d policy groups the network was never trained on:%n", total);
        System.out.println();
        System.out.printf(              "  First %d examples (kSEK):%n", Math.min(EXAMPLES_SHOWN, total));
        System.out.printf("  %10s   %10s   %10s%n", "Actual", "Predicted", "Error");
        for (int i = 0; i < Math.min(EXAMPLES_SHOWN, total); i++) {
            System.out.printf(Locale.ROOT, "  %10.2f   %10.2f   %+10.2f%n", actual[i], predicted[i], predicted[i] - actual[i]);
        }
        System.out.println();

        System.out.printf(Locale.ROOT, "  MAE       %.2f kSEK  - on average the prediction is off by this much%n", mae);
        System.out.printf(Locale.ROOT, "  Baseline  %.2f kSEK  - always answering with the average would be off by this much%n", baselineMae);
        System.out.printf(Locale.ROOT, "  RMSE      %.2f kSEK  - like MAE, but big mistakes count more%n", rmse);
        System.out.printf(Locale.ROOT, "  R2        %.3f  - share of the variation explained (1.0 = perfect, 0.0 = no better than the average)%n", r2);
        System.out.println();
    }

    // Prints your policy group and what the network says about it.
    static void prediction(PolicyGroup group, PaymentPrediction prediction) {
        section("4. Your policy group");
        System.out.printf(Locale.ROOT, "  Kilometres class:   %d%n", group.kilometresClass());
        System.out.printf(Locale.ROOT, "  Zone:               %s%n", group.zone().name().toLowerCase(Locale.ROOT));
        System.out.printf(Locale.ROOT, "  Bonus class:        %d%n", group.bonusClass());
        System.out.printf(Locale.ROOT, "  Vehicle make:       %d%n", group.vehicleMake());
        System.out.printf(Locale.ROOT, "  Insured years:      %.1f%n", group.insuredYears());
        System.out.println();
        System.out.printf(Locale.ROOT, "  Result: The network expects a payment of about %.2f kSEK per insured year.%n", prediction.paymentPerInsuredYearKsek());
        System.out.println();
        System.out.println("  Edit YourPolicyGroup.java and run again to explore how the result changes.");
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
