package com.deepnetts.examples.classification.irisflower;

import com.deepnetts.examples.classification.irisflower.dto.Flower;
import com.deepnetts.examples.classification.irisflower.dto.SpeciesMistake;
import com.deepnetts.examples.classification.irisflower.dto.SpeciesPrediction;
import com.deepnetts.examples.classification.irisflower.enums.Species;
import com.deepnetts.examples.classification.irisflower.pipeline.FlowerEncoder;
import com.deepnetts.examples.classification.irisflower.pipeline.SpeciesConfusion;

import java.util.Locale;

// Prints the output, so IrisFlower contains only the workflow.
final class Report {

    // Every section line is this wide, for example "--- 1. Dataset -------...".
    private static final int WIDTH = 80;

    private Report() {
    }

    // Prints how many flowers there are of each species.
    static void dataset(int totalCount, int[] countPerSpecies) {
        section("1. Dataset");
        System.out.printf(Locale.ROOT, "  Total:       %4d flowers%n", totalCount);
        for (Species species : Species.values()) {
            int count = countPerSpecies[species.ordinal()];
            System.out.printf(Locale.ROOT, "  %-12s %4d (%.1f%%)%n", name(species) + ":", count, 100.0 * count / totalCount);
        }
        System.out.printf(              "  Inputs:      %4d numbers per flower (sepal and petal length and width)%n", FlowerEncoder.NUM_INPUTS);
        System.out.println();
    }

    // Prints the results on the test set (flowers the network never saw).
    // The counts are calculated in SpeciesConfusion; this method only prints them.
    static void evaluation(SpeciesConfusion confusion) {
        int total = confusion.total();

        section("3. Evaluation");
        System.out.println();
        System.out.printf(Locale.ROOT, "  Results on %d flowers the network was never trained on:%n", total);
        System.out.println();
        System.out.printf("  Accuracy   %s  - share of flowers the network classified correctly%n", percent(confusion.correct(), total));
        System.out.printf("  Baseline   %s  - always answering with the most common species would be right this often%n",
                percent(confusion.baselineCorrect(), total));
        System.out.println();

        // The table: one row per real species, one column per answer. The diagonal holds the correct answers;
        // every number off the diagonal is a mistake. The last column is how many of that species were found.
        System.out.println("  Real species (rows) and the network's answer (columns):");
        System.out.printf("  %-18s", "");
        for (Species answer : Species.values()) {
            System.out.printf("%12s", name(answer));
        }
        System.out.println("     Correct");
        for (Species actual : Species.values()) {
            System.out.printf("  %-18s", "Real " + name(actual));
            for (Species answer : Species.values()) {
                System.out.printf("%12d", confusion.count(actual, answer));
            }
            System.out.printf("   %s%n", percent(confusion.count(actual, actual), confusion.actualCount(actual)));
        }
        System.out.println();

        SpeciesMistake mistake = confusion.mostCommonMistake();
        System.out.println(mistake == null
                ? "  No mistakes on the test flowers."
                : String.format("  Most common mistake: a %s called %s (%d %s)", name(mistake.actual()), name(mistake.predicted()),
                        mistake.count(), mistake.count() == 1 ? "time" : "times"));
        System.out.println();
    }

    // Prints your flower and what the network says about it.
    static void prediction(Flower flower, SpeciesPrediction prediction) {
        section("4. Your flower");
        System.out.printf(Locale.ROOT, "  Sepal:  %.1f cm long, %.1f cm wide%n", flower.sepalLengthCm(), flower.sepalWidthCm());
        System.out.printf(Locale.ROOT, "  Petal:  %.1f cm long, %.1f cm wide%n", flower.petalLengthCm(), flower.petalWidthCm());
        System.out.println();
        for (Species species : Species.values()) {
            System.out.printf(Locale.ROOT, "  %-12s %5.1f%%%n", name(species) + ":", 100 * prediction.probabilities()[species.ordinal()]);
        }
        System.out.println();
        System.out.printf("  Result: The network says your flower is %s.%n", name(prediction.species()));
        System.out.println();
        System.out.println("  Edit YourFlower.java and run again to explore how the result changes.");
    }

    // SETOSA -> setosa
    private static String name(Species species) {
        return species.name().toLowerCase(Locale.ROOT);
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
