package com.deepnetts.examples.image_classification.parking;

import com.deepnetts.examples.image_classification.parking.dto.ParkingData;
import com.deepnetts.examples.image_classification.parking.dto.SpotPrediction;
import com.deepnetts.examples.image_classification.parking.pipeline.SpotEncoder;

import deepnetts.eval.ConfusionMatrix;

import java.nio.file.Path;
import java.util.Locale;

// Prints the output, so ParkingSpot contains only the workflow.
final class Report {

    // Every section line is this wide, for example "--- 1. Dataset -------...".
    private static final int WIDTH = 80;

    private static final int EXAMPLES_SHOWN = 5;

    private Report() {
    }

    // Prints how many pictures there are in each set, from which part of the day, and how many spots are taken.
    static void dataset(ParkingData data, int trainingBusy, int testBusy, int trainingStepMinutes) {
        int trainingCount = data.trainingSet().size();
        int testCount = data.testSet().size();

        section("1. Dataset");
        System.out.printf(Locale.ROOT, "  Training: %4d pictures (%d busy, %d free), %s to %s, every %d minutes%n",
                trainingCount, trainingBusy, trainingCount - trainingBusy,
                data.firstTrainingCapture(), data.lastTrainingCapture(), trainingStepMinutes);
        System.out.printf(Locale.ROOT, "  Test:     %4d pictures (%d busy, %d free), %s to %s%n",
                testCount, testBusy, testCount - testBusy, data.firstTestCapture(), data.lastTestCapture());
        System.out.printf(Locale.ROOT, "  Image size: %d x %d pixels, %d colors%n",
                SpotEncoder.IMAGE_WIDTH, SpotEncoder.IMAGE_HEIGHT, SpotEncoder.IMAGE_CHANNELS);
        System.out.println();
    }

    // Prints the results on the test set (pictures from the end of the day the network never saw).
    // For every test picture we know the real answer (actual) and the network's answer (the busy probability).
    // The counts come from Deep Netts (ConfusionMatrix); this method only prints them.
    static void evaluation(String[] imageNames, boolean[] actual, float[] busyProbability, float busyThreshold,
                           ConfusionMatrix confusionMatrix) {
        int truePositives = confusionMatrix.getTruePositive();     // busy, and the network said busy
        int falsePositives = confusionMatrix.getFalsePositive();   // free, but the network said busy
        int trueNegatives = confusionMatrix.getTrueNegative();     // free, and the network said free
        int falseNegatives = confusionMatrix.getFalseNegative();   // busy, but the network said free
        int total = truePositives + trueNegatives + falsePositives + falseNegatives;
        // Always answering with the more common class: right on all images of that class.
        int baselineCorrect = Math.max(truePositives + falseNegatives, trueNegatives + falsePositives);

        section("3. Evaluation");
        System.out.println();
        System.out.printf(Locale.ROOT, "  Results on %d pictures the network was never trained on:%n", total);
        System.out.println();
        System.out.printf(              "  First %d examples:%n", Math.min(EXAMPLES_SHOWN, total));
        System.out.printf("  %-22s %-8s %12s   %s%n", "Image", "Actual", "Busy prob.", "Predicted");
        for (int i = 0; i < Math.min(EXAMPLES_SHOWN, total); i++) {
            System.out.printf(Locale.ROOT, "  %-22s %-8s %11.1f%%   %s%n", imageNames[i],
                    actual[i] ? "Busy" : "Free", busyProbability[i] * 100,
                    busyProbability[i] >= busyThreshold ? "Busy" : "Free");
        }
        System.out.println();

        System.out.printf("  %-12s %14s %16s%n", "", "Predicted FREE", "Predicted BUSY");
        System.out.printf("  %-12s %9d (TN) %11d (FP)%n", "Actual FREE", trueNegatives, falsePositives);
        System.out.printf("  %-12s %9d (FN) %11d (TP)%n", "Actual BUSY", falseNegatives, truePositives);
        System.out.println("  TN and TP are right answers, FP is a false alarm, FN is a taken spot called free.");
        System.out.println();
        System.out.printf("  Accuracy   %s  - share of pictures the network classified correctly%n", percent(truePositives + trueNegatives, total));
        System.out.printf("  Baseline   %s  - always answering with the more common class would be right this often%n", percent(baselineCorrect, total));
        System.out.printf("  Precision  %s  - of the spots the network called busy, this many really are%n", percent(truePositives, truePositives + falsePositives));
        System.out.printf("  Recall     %s  - of the busy spots, this many were found by the network%n", percent(truePositives, truePositives + falseNegatives));
        System.out.println();
    }

    // Prints your picture and what the network says about it.
    static void prediction(Path imagePath, SpotPrediction prediction) {
        section("4. Your image");
        System.out.printf(Locale.ROOT, "  Image:             %s%n", imagePath);
        System.out.println();
        System.out.printf(Locale.ROOT, "  Busy probability:  %.1f%%%n", prediction.busyProbability() * 100);
        System.out.printf("  Prediction:        %s%n", prediction.isBusy() ? "BUSY" : "FREE");
        System.out.println();
        System.out.println(prediction.isBusy()
                ? "  Result: The network says the spot is taken."
                : "  Result: The network says the spot is free.");
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
