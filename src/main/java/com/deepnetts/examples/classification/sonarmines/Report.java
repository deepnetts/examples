package com.deepnetts.examples.classification.sonarmines;

import com.deepnetts.examples.classification.sonarmines.dto.MinePrediction;
import com.deepnetts.examples.classification.sonarmines.dto.SonarContact;
import com.deepnetts.examples.classification.sonarmines.pipeline.ContactEncoder;

import deepnetts.eval.ClassificationMetrics;
import deepnetts.eval.ConfusionMatrix;

import java.util.Locale;

// Prints the output, so SonarMines contains only the workflow.
final class Report {

    // Every section line is this wide, for example "--- 1. Dataset -------...".
    private static final int WIDTH = 80;

    private Report() {
    }

    // Prints how many contacts there are and how many are mines.
    static void dataset(int totalCount, int mineCount) {
        int harmlessCount = totalCount - mineCount;

        section("1. Dataset");
        System.out.printf(Locale.ROOT, "  Total:      %4d sonar contacts%n", totalCount);
        System.out.printf(Locale.ROOT, "  Harmless:   %4d (%.1f%%)%n", harmlessCount, 100.0 * harmlessCount / totalCount);
        System.out.printf(Locale.ROOT, "  Mines:      %4d (%.1f%%)%n", mineCount, 100.0 * mineCount / totalCount);
        System.out.printf(              "  Inputs:     %4d numbers per contact (the features the network learns from)%n", ContactEncoder.NUM_INPUTS);
        System.out.println();
    }

    // Prints the results on the test set (contacts the network never saw).
    // TP = mine called mine, TN = harmless object called harmless,
    // FP = harmless object wrongly called a mine, FN = mine wrongly called harmless.
    static void evaluation(ConfusionMatrix cm, ClassificationMetrics metrics, float threshold) {
        int tp = cm.getTruePositive();
        int tn = cm.getTrueNegative();
        int fp = cm.getFalsePositive();
        int fn = cm.getFalseNegative();
        int total  = tp + tn + fp + fn;
        int errors = fp + fn;
        // The score of a "network" that always guesses the more common answer. Ours must beat this.
        int baseline = Math.max(tp + fn, tn + fp);

        section("3. Evaluation");
        System.out.println();
        System.out.printf(Locale.ROOT, "  Results on %d contacts the network was never trained on:%n", total);
        System.out.printf(Locale.ROOT, "  Threshold: %.2f - a probability at or above it counts as MINE%n", threshold);
        System.out.println();
        System.out.printf(Locale.ROOT, "  Caught correctly:  %3d mines  + %d harmless = %d correct  (%.1f%%)%n",
                tp, tn, tp + tn, 100.0 * (tp + tn) / total);
        System.out.printf(Locale.ROOT, "  Got wrong:         %3d mines missed + %d false alarms = %d errors (%.1f%%)%n",
                fn, fp, errors, 100.0 * errors / total);
        System.out.println();
        System.out.printf(              "  Of the %d errors:%n", errors);
        System.out.printf(              "    %d missed mines   - real mines the network called harmless%n", fn);
        System.out.printf(              "    %d false alarms   - harmless objects wrongly called mines%n", fp);
        System.out.println();

        System.out.printf( "                    Predicted HARMLESS   Predicted MINE%n");
        System.out.printf( "  Actual HARMLESS %8d (TN)       %8d (FP)%n", tn, fp);
        System.out.printf( "  Actual MINE     %8d (FN)       %8d (TP)%n", fn, tp);
        System.out.println("  TN and TP are right answers, FP is a false alarm, FN is a missed mine.");
        System.out.println();

        System.out.printf(Locale.ROOT, "  Accuracy  %.1f%%  - %d of %d correct%n",
                100 * metrics.getAccuracy(), tp + tn, total);
        System.out.printf(Locale.ROOT, "  Baseline  %.1f%%  - always guessing the more common answer would be this accurate%n",
                100.0 * baseline / total);
        System.out.printf(Locale.ROOT, "  Precision %.1f%%  - when called MINE, it really is a mine (%d of %d flagged)%n",
                100 * metrics.getPrecision(), tp, tp + fp);
        System.out.printf(Locale.ROOT, "  Recall    %.1f%%  - %d of the %d actual mines were caught%n",
                100 * metrics.getRecall(), tp, tp + fn);
        System.out.printf(Locale.ROOT, "  F1        %.3f  - combined precision+recall score (1.0 = perfect)%n",
                metrics.getF1Score());
        System.out.println();
    }

    // Prints your sonar contact and what the network says about it.
    static void prediction(SonarContact contact, MinePrediction prediction) {
        section("4. Your sonar contact");
        System.out.printf(Locale.ROOT, "  Range:              %.1f m%n", contact.rangeM());
        System.out.printf(              "  Aspect angle:       %d degrees%n", contact.aspectAngleDeg());
        System.out.printf(Locale.ROOT, "  Echo level:         %.1f dB%n", contact.echoLevelDb());
        System.out.printf(Locale.ROOT, "  Echo duration:      %.2f ms%n", contact.echoDurationMs());
        System.out.printf(Locale.ROOT, "  Spectral centroid:  %.1f kHz%n", contact.spectralCentroidKhz());
        System.out.printf(Locale.ROOT, "  Shadow length:      %.1f m%n", contact.shadowLengthM());
        System.out.printf(              "  Seabed:             %s%n", contact.seabedType().name().toLowerCase(Locale.ROOT));
        System.out.println();
        System.out.printf(Locale.ROOT, "  Mine probability:   %.1f%%%n", 100 * prediction.mineProbability());
        System.out.printf(              "  Prediction:         %s%n",
                prediction.isMine() ? "MINE" : "HARMLESS");
        System.out.println();
        System.out.println(prediction.isMine()
                ? "  Result: The network flagged your contact as a mine."
                : "  Result: The network considers your contact harmless.");
        System.out.println();
        System.out.println("  Edit YourContact.java and run again to explore how the result changes.");
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
