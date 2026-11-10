package com.deepnetts.examples.classification.creditcardfraud;

import com.deepnetts.examples.classification.creditcardfraud.dto.FraudPrediction;
import com.deepnetts.examples.classification.creditcardfraud.dto.Transaction;
import com.deepnetts.examples.classification.creditcardfraud.pipeline.TransactionEncoder;

import deepnetts.eval.ClassificationMetrics;
import deepnetts.eval.ConfusionMatrix;

import java.util.Locale;

// Prints the output, so CreditCardFraud contains only the workflow.
final class Report {

    // Every section line is this wide, for example "--- 1. Dataset -------...".
    private static final int WIDTH = 80;

    private Report() {
    }

    // Prints how many transactions there are and how many are fraud.
    static void dataset(int totalCount, int fraudCount) {
        int legitimateCount = totalCount - fraudCount;

        section("1. Dataset");
        System.out.printf(Locale.ROOT, "  Total:      %4d transactions%n", totalCount);
        System.out.printf(Locale.ROOT, "  Legitimate: %4d (%.1f%%)%n", legitimateCount, 100.0 * legitimateCount / totalCount);
        System.out.printf(Locale.ROOT, "  Fraud:      %4d (%.1f%%)%n", fraudCount, 100.0 * fraudCount / totalCount);
        System.out.printf(              "  Inputs:     %4d numbers per transaction (the features the network learns from)%n", TransactionEncoder.NUM_INPUTS);
        System.out.println();
    }

    // Prints the results on the test set (transactions the network never saw).
    static void evaluation(ConfusionMatrix confusionMatrix, ClassificationMetrics metrics, float threshold) {
        int tp = confusionMatrix.getTruePositive();    // fraud, and the network said fraud
        int tn = confusionMatrix.getTrueNegative();    // legitimate, and the network said legitimate
        int fp = confusionMatrix.getFalsePositive();   // legitimate, but the network said fraud (false alarm)
        int fn = confusionMatrix.getFalseNegative();   // fraud, but the network missed it
        int total  = tp + tn + fp + fn;
        int errors = fp + fn;
        int baseline = Math.max(tp + fn, tn + fp);   // the accuracy of always answering with the more common class

        section("3. Evaluation");
        System.out.println();
        System.out.printf(Locale.ROOT, "  Results on %d transactions the network was never trained on:%n", total);
        System.out.printf(Locale.ROOT, "  Threshold: %.2f - a probability at or above it counts as FRAUD%n", threshold);
        System.out.println();
        System.out.printf(Locale.ROOT, "  Caught correctly:  %3d fraud  + %d legitimate = %d correct  (%.1f%%)%n",
                tp, tn, tp + tn, 100.0 * (tp + tn) / total);
        System.out.printf(Locale.ROOT, "  Got wrong:         %3d fraud missed + %d false alarms = %d errors (%.1f%%)%n",
                fn, fp, errors, 100.0 * errors / total);
        System.out.println();
        System.out.printf(              "  Of the %d errors:%n", errors);
        System.out.printf(              "    %d missed fraud   - real fraud the network called legitimate%n", fn);
        System.out.printf(              "    %d false alarms   - legitimate transactions wrongly flagged%n", fp);
        System.out.println();

        System.out.printf("  %-18s %20s %17s%n", "", "Predicted LEGITIMATE", "Predicted FRAUD");
        System.out.printf("  %-18s %15d (TN) %12d (FP)%n", "Actual LEGITIMATE", tn, fp);
        System.out.printf("  %-18s %15d (FN) %12d (TP)%n", "Actual FRAUD", fn, tp);
        System.out.println("  TN and TP are right answers, FP is a false alarm, FN is a missed fraud.");
        System.out.println();

        System.out.printf(Locale.ROOT, "  Accuracy  %.1f%%  - %d of %d correct%n",
                100 * metrics.getAccuracy(), tp + tn, total);
        System.out.printf(Locale.ROOT, "  Baseline  %.1f%%  - always guessing the more common answer would be this accurate%n",
                100.0 * baseline / total);
        System.out.printf(Locale.ROOT, "  Precision %.1f%%  - when flagged as FRAUD, it really is fraud (%d of %d flagged)%n",
                100 * metrics.getPrecision(), tp, tp + fp);
        System.out.printf(Locale.ROOT, "  Recall    %.1f%%  - %d of the %d actual fraud cases were caught%n",
                100 * metrics.getRecall(), tp, tp + fn);
        System.out.printf(Locale.ROOT, "  F1        %.3f  - combined precision+recall score (1.0 = perfect)%n",
                metrics.getF1Score());
        System.out.println();
    }

    // Prints your transaction and what the network says about it.
    static void prediction(Transaction transaction, FraudPrediction prediction) {
        section("4. Your transaction");
        System.out.printf(Locale.ROOT, "  Amount:             %.2f%n",  transaction.amount());
        System.out.printf(              "  Category:           %s%n",    transaction.merchantCategory().name().toLowerCase(Locale.ROOT));
        System.out.printf(              "  Time:               %02d:00%n", transaction.hourOfDay());
        System.out.printf(Locale.ROOT, "  Distance from home: %.1f km%n", transaction.distanceFromHomeKm());
        System.out.println();
        System.out.printf(Locale.ROOT, "  Fraud probability:  %.1f%%%n", 100 * prediction.fraudProbability());
        System.out.printf(              "  Prediction:         %s%n",
                prediction.isFraud() ? "FRAUD" : "LEGITIMATE");
        System.out.println();
        System.out.println(prediction.isFraud()
                ? "  Result: The network flagged your transaction as fraud."
                : "  Result: The network considers your transaction legitimate.");
        System.out.println();
        System.out.println("  Edit YourTransaction.java and run again to explore how the result changes.");
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
