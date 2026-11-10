package com.deepnetts.examples.image_classification.digits;

import com.deepnetts.examples.image_classification.digits.dto.DigitMistake;
import com.deepnetts.examples.image_classification.digits.dto.DigitPrediction;
import com.deepnetts.examples.image_classification.digits.pipeline.DigitConfusion;
import com.deepnetts.examples.image_classification.digits.pipeline.DigitEncoder;

import java.nio.file.Path;
import java.util.Locale;

// Prints the output, so Digits contains only the workflow.
final class Report {

    // Every section line is this wide, for example "--- 1. Dataset -------...".
    private static final int WIDTH = 80;

    private static final int EXAMPLES_SHOWN = 5;

    private Report() {
    }

    // Prints how many images there are of each digit.
    static void dataset(int totalCount, int[] countPerDigit) {
        section("1. Dataset");
        System.out.printf(Locale.ROOT, "  Total:      %4d images%n", totalCount);
        System.out.print("  Per digit: ");
        for (int digit = 0; digit < countPerDigit.length; digit++) {
            System.out.printf(Locale.ROOT, " %d:%d", digit, countPerDigit[digit]);
        }
        System.out.println();
        System.out.printf(Locale.ROOT, "  Image size: %d x %d pixels, gray%n", DigitEncoder.IMAGE_WIDTH, DigitEncoder.IMAGE_HEIGHT);
        System.out.println();
    }

    // Prints the results on the test set (images the network never saw).
    // For every test image we know the real digit (actual), the network's answer (predicted)
    // and how sure the network was. The counts are calculated in DigitConfusion; this method only prints them.
    static void evaluation(String[] imageNames, int[] actual, int[] predicted, float[] probability, DigitConfusion confusion) {
        int total = confusion.total();

        section("3. Evaluation");
        System.out.println();
        System.out.printf(Locale.ROOT, "  Results on %d images the network was never trained on:%n", total);
        System.out.println();
        System.out.printf(              "  First %d examples:%n", Math.min(EXAMPLES_SHOWN, total));
        System.out.printf("  %-12s %-8s %-10s %12s%n", "Image", "Actual", "Predicted", "Probability");
        for (int i = 0; i < Math.min(EXAMPLES_SHOWN, total); i++) {
            System.out.printf(Locale.ROOT, "  %-12s %-8d %-10d %11.1f%%%n", imageNames[i], actual[i], predicted[i], probability[i] * 100);
        }
        System.out.println();

        System.out.printf("  Accuracy   %s  - share of images the network read correctly%n", percent(confusion.correct(), total));
        System.out.printf("  Baseline   %s  - always answering with the most common digit would be right this often%n",
                percent(confusion.baselineCorrect(), total));
        System.out.println();

        // The table: one row per real digit, one column per answer. The diagonal holds the correct answers;
        // every number off the diagonal is a mistake. The last column is how many of that digit were read correctly.
        System.out.println("  Real digit (rows) and the network's answer (columns):");
        System.out.print("            ");
        for (int digit = 0; digit < DigitEncoder.NUM_OUTPUTS; digit++) {
            System.out.printf("%4d", digit);
        }
        System.out.println("     Correct");
        for (int actualDigit = 0; actualDigit < DigitEncoder.NUM_OUTPUTS; actualDigit++) {
            System.out.printf("  Real %d    ", actualDigit);
            for (int answer = 0; answer < DigitEncoder.NUM_OUTPUTS; answer++) {
                int count = confusion.count(actualDigit, answer);
                System.out.print(count == 0 ? "   ." : String.format("%4d", count));
            }
            System.out.printf("   %s%n", percent(confusion.count(actualDigit, actualDigit), confusion.actualCount(actualDigit)));
        }
        System.out.println();

        DigitMistake mistake = confusion.mostCommonMistake();
        if (mistake.count() > 0) {
            System.out.printf("  Most common mistake: a %d read as a %d (%d %s)%n", mistake.actual(), mistake.predicted(),
                    mistake.count(), mistake.count() == 1 ? "time" : "times");
            System.out.println();
        }
    }

    // Prints your image and what the network says about it.
    static void prediction(Path imagePath, DigitPrediction prediction) {
        section("4. Your image");
        System.out.printf(Locale.ROOT, "  Image:             %s%n", imagePath);
        System.out.println();
        System.out.printf(Locale.ROOT, "  Prediction:        %d%n", prediction.digit());
        System.out.printf(Locale.ROOT, "  Probability:       %.1f%%%n", prediction.probability() * 100);
        System.out.println();
        System.out.printf(Locale.ROOT, "  Result: The network reads your image as a %d.%n", prediction.digit());
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
