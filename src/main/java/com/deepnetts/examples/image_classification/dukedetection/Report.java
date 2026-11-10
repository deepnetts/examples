package com.deepnetts.examples.image_classification.dukedetection;

import com.deepnetts.examples.image_classification.dukedetection.dto.DukePrediction;
import com.deepnetts.examples.image_classification.dukedetection.pipeline.DukeEncoder;

import deepnetts.eval.ConfusionMatrix;

import java.nio.file.Path;
import java.util.Locale;

// Prints the output, so DukeDetection contains only the workflow.
final class Report {

    // Every section line is this wide, for example "--- 1. Dataset -------...".
    private static final int WIDTH = 80;

    private static final int EXAMPLES_SHOWN = 5;

    private Report() {
    }

    // Prints how many images there are and how many of them show Duke.
    static void dataset(int totalCount, int dukeCount) {
        section("1. Dataset");
        System.out.printf(Locale.ROOT, "  Total:      %4d images%n", totalCount);
        System.out.printf(Locale.ROOT, "  Duke:       %4d images%n", dukeCount);
        System.out.printf(Locale.ROOT, "  Not Duke:   %4d images%n", totalCount - dukeCount);
        System.out.printf(Locale.ROOT, "  Image size: %d x %d pixels, %d colors%n",
                DukeEncoder.IMAGE_WIDTH, DukeEncoder.IMAGE_HEIGHT, DukeEncoder.IMAGE_CHANNELS);
        System.out.println();
    }

    // Prints the results on the test set (images the network never saw).
    // For every test image we know the real answer (actual) and the network's answer (the Duke probability).
    // The counts come from Deep Netts (ConfusionMatrix); this method only prints them.
    static void evaluation(String[] imageNames, boolean[] actual, float[] dukeProbability, float dukeThreshold,
                           ConfusionMatrix confusionMatrix) {
        int truePositives = confusionMatrix.getTruePositive();     // Duke, and the network said Duke
        int falsePositives = confusionMatrix.getFalsePositive();   // not Duke, but the network said Duke
        int trueNegatives = confusionMatrix.getTrueNegative();     // not Duke, and the network said not Duke
        int falseNegatives = confusionMatrix.getFalseNegative();   // Duke, but the network said not Duke
        int total = truePositives + trueNegatives + falsePositives + falseNegatives;
        // Always answering with the more common class: right on all images of that class.
        int baselineCorrect = Math.max(truePositives + falseNegatives, trueNegatives + falsePositives);

        section("3. Evaluation");
        System.out.println();
        System.out.printf(Locale.ROOT, "  Results on %d images the network was never trained on:%n", total);
        System.out.println();
        System.out.printf(              "  First %d examples:%n", Math.min(EXAMPLES_SHOWN, total));
        System.out.printf("  %-16s %-10s %14s   %s%n", "Image", "Actual", "Duke probability", "Predicted");
        for (int i = 0; i < Math.min(EXAMPLES_SHOWN, total); i++) {
            System.out.printf(Locale.ROOT, "  %-16s %-10s %15.1f%%   %s%n", imageNames[i],
                    actual[i] ? "Duke" : "Not Duke", dukeProbability[i] * 100,
                    dukeProbability[i] >= dukeThreshold ? "Duke" : "Not Duke");
        }
        System.out.println();

        System.out.printf("  %-16s %18s %16s%n", "", "Predicted NOT DUKE", "Predicted DUKE");
        System.out.printf("  %-16s %13d (TN) %11d (FP)%n", "Actual NOT DUKE", trueNegatives, falsePositives);
        System.out.printf("  %-16s %13d (FN) %11d (TP)%n", "Actual DUKE", falseNegatives, truePositives);
        System.out.println("  TN and TP are right answers, FP is a false alarm, FN is a missed Duke image.");
        System.out.println();
        System.out.printf("  Accuracy   %s  - share of images the network classified correctly%n", percent(truePositives + trueNegatives, total));
        System.out.printf("  Baseline   %s  - always answering with the more common class would be right this often%n", percent(baselineCorrect, total));
        System.out.printf("  Precision  %s  - of the images the network called Duke, this many really are Duke%n", percent(truePositives, truePositives + falsePositives));
        System.out.printf("  Recall     %s  - of the real Duke images, this many were found by the network%n", percent(truePositives, truePositives + falseNegatives));
        System.out.println();
    }

    // Prints your image and what the network says about it.
    static void prediction(Path imagePath, DukePrediction prediction) {
        section("4. Your image");
        System.out.printf(Locale.ROOT, "  Image:              %s%n", imagePath);
        System.out.println();
        System.out.printf(Locale.ROOT, "  Duke probability:   %.1f%%%n", prediction.dukeProbability() * 100);
        System.out.printf("  Prediction:         %s%n", prediction.isDuke() ? "DUKE" : "NOT DUKE");
        System.out.println();
        System.out.println(prediction.isDuke()
                ? "  Result: The network says your image shows Duke."
                : "  Result: The network says your image does not show Duke.");
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
