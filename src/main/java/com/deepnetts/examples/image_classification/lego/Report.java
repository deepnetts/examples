package com.deepnetts.examples.image_classification.lego;

import com.deepnetts.examples.image_classification.lego.dto.LegoPrediction;
import com.deepnetts.examples.image_classification.lego.pipeline.LegoEncoder;

import deepnetts.eval.ConfusionMatrix;

import java.nio.file.Path;
import java.util.Locale;

// Prints the output, so LegoFigure contains only the workflow.
final class Report {

    // Every section line is this wide, for example "--- 1. Dataset -------...".
    private static final int WIDTH = 80;

    private static final int EXAMPLES_SHOWN = 5;

    private Report() {
    }

    // Prints how many images there are and how many of them show a Lego figure.
    static void dataset(int totalCount, int legoCount) {
        section("1. Dataset");
        System.out.printf(Locale.ROOT, "  Total:       %4d images%n", totalCount);
        System.out.printf(Locale.ROOT, "  Lego figure: %4d images%n", legoCount);
        System.out.printf(Locale.ROOT, "  Other:       %4d images (Duke, food and parking pictures)%n", totalCount - legoCount);
        System.out.printf(Locale.ROOT, "  Image size:  %d x %d pixels, %d colors%n",
                LegoEncoder.IMAGE_WIDTH, LegoEncoder.IMAGE_HEIGHT, LegoEncoder.IMAGE_CHANNELS);
        System.out.println();
    }

    // Prints the results on the test set (images the network never saw).
    // For every test image we know the real answer (actual) and the network's answer (the Lego figure probability).
    // The counts come from Deep Netts (ConfusionMatrix); this method only prints them.
    static void evaluation(String[] imageNames, boolean[] actual, float[] legoProbability, float legoThreshold,
                           ConfusionMatrix confusionMatrix) {
        int truePositives = confusionMatrix.getTruePositive();     // Lego figure, and the network said Lego figure
        int falsePositives = confusionMatrix.getFalsePositive();   // something else, but the network said Lego figure
        int trueNegatives = confusionMatrix.getTrueNegative();     // something else, and the network said something else
        int falseNegatives = confusionMatrix.getFalseNegative();   // Lego figure, but the network said something else
        int total = truePositives + trueNegatives + falsePositives + falseNegatives;
        // Always answering with the more common class: right on all images of that class.
        int baselineCorrect = Math.max(truePositives + falseNegatives, trueNegatives + falsePositives);

        section("3. Evaluation");
        System.out.println();
        System.out.printf(Locale.ROOT, "  Results on %d images the network was never trained on:%n", total);
        System.out.println();
        System.out.printf(              "  First %d examples:%n", Math.min(EXAMPLES_SHOWN, total));
        System.out.printf("  %-16s %-10s %14s   %s%n", "Image", "Actual", "Lego prob.", "Predicted");
        for (int i = 0; i < Math.min(EXAMPLES_SHOWN, total); i++) {
            System.out.printf(Locale.ROOT, "  %-16s %-10s %15.1f%%   %s%n", imageNames[i],
                    actual[i] ? "Lego" : "Other", legoProbability[i] * 100,
                    legoProbability[i] >= legoThreshold ? "Lego" : "Other");
        }
        System.out.println();

        System.out.printf("  %-13s %15s %16s%n", "", "Predicted OTHER", "Predicted LEGO");
        System.out.printf("  %-13s %10d (TN) %11d (FP)%n", "Actual OTHER", trueNegatives, falsePositives);
        System.out.printf("  %-13s %10d (FN) %11d (TP)%n", "Actual LEGO", falseNegatives, truePositives);
        System.out.println("  TN and TP are right answers, FP is a false alarm, FN is a missed Lego figure.");
        System.out.println();
        System.out.printf("  Accuracy   %s  - share of images the network classified correctly%n", percent(truePositives + trueNegatives, total));
        System.out.printf("  Baseline   %s  - always answering with the more common class would be right this often%n", percent(baselineCorrect, total));
        System.out.printf("  Precision  %s  - of the images the network called a Lego figure, this many really are%n", percent(truePositives, truePositives + falsePositives));
        System.out.printf("  Recall     %s  - of the real Lego figures, this many were found by the network%n", percent(truePositives, truePositives + falseNegatives));
        System.out.println();
    }

    // Prints your image and what the network says about it.
    static void prediction(Path imagePath, LegoPrediction prediction) {
        section("4. Your image");
        System.out.printf(Locale.ROOT, "  Image:             %s%n", imagePath);
        System.out.println();
        System.out.printf(Locale.ROOT, "  Lego probability:  %.1f%%%n", prediction.legoProbability() * 100);
        System.out.printf("  Prediction:        %s%n", prediction.isLego() ? "LEGO" : "OTHER");
        System.out.println();
        System.out.println(prediction.isLego()
                ? "  Result: The network says your picture shows a Lego figure."
                : "  Result: The network says your picture does not show a Lego figure.");
        System.out.println();
        System.out.println("  Edit YourImage.java and run again to explore how the result changes.");
    }

    // A share of a whole as a percent, or "n/a" when there is nothing to divide by.
    private static String percent(int part, int whole) {
        if (whole == 0) {
            return "   n/a";
        }
        return String.format(Locale.ROOT, "%5.1f%%", 100.0 * part / whole);
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
