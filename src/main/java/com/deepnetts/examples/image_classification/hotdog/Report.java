package com.deepnetts.examples.image_classification.hotdog;

import com.deepnetts.examples.image_classification.hotdog.dto.HotDogPrediction;
import com.deepnetts.examples.image_classification.hotdog.pipeline.HotDogEncoder;

import deepnetts.eval.ConfusionMatrix;

import java.nio.file.Path;
import java.util.Locale;

// Prints the output, so HotDog contains only the workflow.
final class Report {

    // Every section line is this wide, for example "--- 1. Dataset -------...".
    private static final int WIDTH = 80;

    private static final int EXAMPLES_SHOWN = 5;

    private Report() {
    }

    // Prints how many images there are and how many of them show a hot dog.
    static void dataset(int totalCount, int hotDogCount) {
        section("1. Dataset");
        System.out.printf(Locale.ROOT, "  Total:      %4d images%n", totalCount);
        System.out.printf(Locale.ROOT, "  Hot dog:    %4d images%n", hotDogCount);
        System.out.printf(Locale.ROOT, "  Other food: %4d images%n", totalCount - hotDogCount);
        System.out.printf(Locale.ROOT, "  Image size: %d x %d pixels, %d colors%n",
                HotDogEncoder.IMAGE_WIDTH, HotDogEncoder.IMAGE_HEIGHT, HotDogEncoder.IMAGE_CHANNELS);
        System.out.println();
    }

    // Prints the results on the test set (images the network never saw).
    // For every test image we know the real answer (actual) and the network's answer (the hot dog probability).
    // The counts come from Deep Netts (ConfusionMatrix); this method only prints them.
    static void evaluation(String[] imageNames, boolean[] actual, float[] hotDogProbability, float hotDogThreshold,
                           ConfusionMatrix confusionMatrix) {
        int truePositives = confusionMatrix.getTruePositive();     // hot dog, and the network said hot dog
        int falsePositives = confusionMatrix.getFalsePositive();   // other food, but the network said hot dog
        int trueNegatives = confusionMatrix.getTrueNegative();     // other food, and the network said other food
        int falseNegatives = confusionMatrix.getFalseNegative();   // hot dog, but the network said other food
        int total = truePositives + trueNegatives + falsePositives + falseNegatives;
        // Always answering with the more common class: right on all images of that class.
        int baselineCorrect = Math.max(truePositives + falseNegatives, trueNegatives + falsePositives);

        section("3. Evaluation");
        System.out.println();
        System.out.printf(Locale.ROOT, "  Results on %d images the network was never trained on:%n", total);
        System.out.println();
        System.out.printf(              "  First %d examples:%n", Math.min(EXAMPLES_SHOWN, total));
        System.out.printf("  %-16s %-10s %14s   %s%n", "Image", "Actual", "Hot dog prob.", "Predicted");
        for (int i = 0; i < Math.min(EXAMPLES_SHOWN, total); i++) {
            System.out.printf(Locale.ROOT, "  %-16s %-10s %15.1f%%   %s%n", imageNames[i],
                    actual[i] ? "Hot dog" : "Other", hotDogProbability[i] * 100,
                    hotDogProbability[i] >= hotDogThreshold ? "Hot dog" : "Other");
        }
        System.out.println();

        System.out.printf("  %-15s %15s %19s%n", "", "Predicted OTHER", "Predicted HOT DOG");
        System.out.printf("  %-15s %10d (TN) %14d (FP)%n", "Actual OTHER", trueNegatives, falsePositives);
        System.out.printf("  %-15s %10d (FN) %14d (TP)%n", "Actual HOT DOG", falseNegatives, truePositives);
        System.out.println("  TN and TP are right answers, FP is a false alarm, FN is a missed hot dog.");
        System.out.println();
        System.out.printf("  Accuracy   %s  - share of images the network classified correctly%n", percent(truePositives + trueNegatives, total));
        System.out.printf("  Baseline   %s  - always answering with the more common class would be right this often%n", percent(baselineCorrect, total));
        System.out.printf("  Precision  %s  - of the images the network called a hot dog, this many really are%n", percent(truePositives, truePositives + falsePositives));
        System.out.printf("  Recall     %s  - of the real hot dogs, this many were found by the network%n", percent(truePositives, truePositives + falseNegatives));
        System.out.println();

        // A result not far above the baseline means the network has not learned the task: say it plainly.
        if (truePositives + trueNegatives < baselineCorrect + 0.15 * total) {
            System.out.println("  Note: this is only a little better than the baseline. A small network trained from scratch on a few hundred");
            System.out.println("  photos does not learn what a hot dog looks like. The VGG16 example uses a network that was already trained");
            System.out.println("  on 1.2 million photos, and its 1000 categories include hotdog.");
            System.out.println();
        }
    }

    // Prints your image and what the network says about it.
    static void prediction(Path imagePath, HotDogPrediction prediction) {
        section("4. Your image");
        System.out.printf(Locale.ROOT, "  Image:               %s%n", imagePath);
        System.out.println();
        System.out.printf(Locale.ROOT, "  Hot dog probability: %.1f%%%n", prediction.hotDogProbability() * 100);
        System.out.printf("  Prediction:          %s%n", prediction.isHotDog() ? "HOT DOG" : "OTHER");
        System.out.println();
        System.out.println(prediction.isHotDog()
                ? "  Result: The network says your photo shows a hot dog."
                : "  Result: The network says your photo does not show a hot dog.");
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
