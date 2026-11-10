package com.deepnetts.examples.classification.spamemail;

import com.deepnetts.examples.classification.spamemail.dto.Email;
import com.deepnetts.examples.classification.spamemail.dto.SpamPrediction;
import com.deepnetts.examples.classification.spamemail.pipeline.EmailEncoder;

import deepnetts.eval.ClassificationMetrics;
import deepnetts.eval.ConfusionMatrix;

import java.util.Locale;

// Prints the output, so SpamEmail contains only the workflow.
final class Report {

    // Every section line is this wide, for example "--- 1. Dataset -------...".
    private static final int WIDTH = 80;

    private Report() {
    }

    // Prints how many emails there are and how many are spam.
    static void dataset(int totalCount, int spamCount) {
        int legitCount = totalCount - spamCount;

        section("1. Dataset");
        System.out.printf(Locale.ROOT, "  Total:      %4d emails%n", totalCount);
        System.out.printf(Locale.ROOT, "  Legitimate: %4d (%.1f%%)%n", legitCount, 100.0 * legitCount / totalCount);
        System.out.printf(Locale.ROOT, "  Spam:       %4d (%.1f%%)%n", spamCount, 100.0 * spamCount / totalCount);
        System.out.printf(              "  Inputs:     %4d numbers per email (the features the network learns from)%n", EmailEncoder.NUM_INPUTS);
        System.out.println();
    }

    // Prints the results on the test set (emails the network never saw).
    // TP = spam called spam, TN = legitimate email called legitimate,
    // FP = legitimate email wrongly called spam, FN = spam wrongly called legitimate.
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
        System.out.printf(Locale.ROOT, "  Results on %d emails the network was never trained on:%n", total);
        System.out.printf(Locale.ROOT, "  Threshold: %.2f - a probability at or above it counts as SPAM%n", threshold);
        System.out.println();
        System.out.printf(Locale.ROOT, "  Caught correctly:  %3d spam  + %d legitimate = %d correct  (%.1f%%)%n",
                tp, tn, tp + tn, 100.0 * (tp + tn) / total);
        System.out.printf(Locale.ROOT, "  Got wrong:         %3d spam missed + %d false alarms = %d errors (%.1f%%)%n",
                fn, fp, errors, 100.0 * errors / total);
        System.out.println();
        System.out.printf(              "  Of the %d errors:%n", errors);
        System.out.printf(              "    %d missed spam    - real spam the network called legitimate%n", fn);
        System.out.printf(              "    %d false alarms   - legitimate emails wrongly flagged as spam%n", fp);
        System.out.println();

        System.out.printf( "                    Predicted LEGIT   Predicted SPAM%n");
        System.out.printf( "  Actual LEGIT    %8d (TN)    %8d (FP)%n", tn, fp);
        System.out.printf( "  Actual SPAM     %8d (FN)    %8d (TP)%n", fn, tp);
        System.out.println("  TN and TP are right answers, FP is a false alarm, FN is a missed spam email.");
        System.out.println();

        System.out.printf(Locale.ROOT, "  Accuracy  %.1f%%  - %d of %d correct%n",
                100 * metrics.getAccuracy(), tp + tn, total);
        System.out.printf(Locale.ROOT, "  Baseline  %.1f%%  - always guessing the more common answer would be this accurate%n",
                100.0 * baseline / total);
        System.out.printf(Locale.ROOT, "  Precision %.1f%%  - when flagged as SPAM, it really is spam (%d of %d flagged)%n",
                100 * metrics.getPrecision(), tp, tp + fp);
        System.out.printf(Locale.ROOT, "  Recall    %.1f%%  - %d of the %d actual spam emails were caught%n",
                100 * metrics.getRecall(), tp, tp + fn);
        System.out.printf(Locale.ROOT, "  F1        %.3f  - combined precision+recall score (1.0 = perfect)%n",
                metrics.getF1Score());
        System.out.println();
    }

    // Prints your email and what the network says about it.
    static void prediction(Email email, SpamPrediction prediction) {
        section("4. Your email");
        System.out.printf(Locale.ROOT, "  \"free\" word:        %.2f%%%n", email.freeWordPercent());
        System.out.printf(Locale.ROOT, "  \"money\" word:       %.2f%%%n", email.moneyWordPercent());
        System.out.printf(Locale.ROOT, "  \"remove\" word:      %.2f%%%n", email.removeWordPercent());
        System.out.printf(Locale.ROOT, "  \"your\" word:        %.2f%%%n", email.yourWordPercent());
        System.out.printf(Locale.ROOT, "  \"!\" characters:     %.2f%%%n", email.exclamationCharPercent());
        System.out.printf(Locale.ROOT, "  \"$\" characters:     %.2f%%%n", email.dollarCharPercent());
        System.out.printf(              "  Longest capital run: %d%n", email.longestCapitalRun());
        System.out.printf(              "  Total capitals:      %d%n", email.totalCapitalLetters());
        System.out.println();
        System.out.printf(Locale.ROOT, "  Spam probability:   %.1f%%%n", 100 * prediction.spamProbability());
        System.out.printf(              "  Prediction:         %s%n",
                prediction.isSpam() ? "SPAM" : "LEGITIMATE");
        System.out.println();
        System.out.println(prediction.isSpam()
                ? "  Result: The network flagged your email as spam."
                : "  Result: The network considers your email legitimate.");
        System.out.println();
        System.out.println("  Edit YourEmail.java and run again to explore how the result changes.");
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
