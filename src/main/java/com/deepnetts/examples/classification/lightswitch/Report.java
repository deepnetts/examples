package com.deepnetts.examples.classification.lightswitch;

import com.deepnetts.examples.classification.lightswitch.dto.LightPrediction;
import com.deepnetts.examples.classification.lightswitch.dto.Switches;
import com.deepnetts.examples.classification.lightswitch.dto.TruthTableRow;

import java.util.List;
import java.util.Locale;

// Prints the output, so LightSwitches contains only the workflow.
final class Report {

    // Every section line is this wide, for example "--- 1. Dataset -------...".
    private static final int WIDTH = 80;

    // At or above this probability, a network's answer counts as "on".
    private static final float ON_THRESHOLD = 0.5f;

    // An answer this close to 50% (between 45% and 55%) is not really an answer: the network can't decide.
    private static final float UNSURE_MARGIN = 0.05f;

    private Report() {
    }

    // Prints how many positions of the switches there are.
    static void dataset(int totalCount) {
        section("1. Dataset");
        System.out.printf(Locale.ROOT, "  Total:      %d positions of the two switches (all there are)%n", totalCount);
        System.out.println("  Rule:       the light is on when exactly one switch is up (XOR)");
        System.out.println();
    }

    // Prints when the training starts.
    static void trainingStarted() {
        section("2. Training");
        System.out.println("  Training two networks on the same 4 rows...");
    }

    // Prints how many epochs the training of one network took.
    static void trainingFinished(String network, int epochs) {
        System.out.printf("  %-24s stopped after %d epochs%n", network + ":", epochs);
    }

    // Prints what both networks say about every position of the switches.
    static void evaluation(List<TruthTableRow> rows) {
        int correctWithout = 0;
        int correctWith = 0;
        int unsureWithout = 0;
        int unsureWith = 0;
        System.out.println();
        section("3. Evaluation");
        System.out.println();
        System.out.printf("  %-8s %-8s %-7s %-24s %s%n", "Bottom", "Top", "Light", "Without a hidden layer", "With a hidden layer");
        for (TruthTableRow row : rows) {
            unsureWithout += isUnsure(row.withoutHiddenLayer()) ? 1 : 0;
            unsureWith += isUnsure(row.withHiddenLayer()) ? 1 : 0;
            correctWithout += isRight(row.withoutHiddenLayer(), row.lightOn()) ? 1 : 0;
            correctWith += isRight(row.withHiddenLayer(), row.lightOn()) ? 1 : 0;
            System.out.printf(Locale.ROOT, "  %-8s %-8s %-7s %-24s %s%n",
                    upOrDown(row.bottomUp()), upOrDown(row.topUp()), onOrOff(row.lightOn()),
                    answer(row.withoutHiddenLayer(), row.lightOn()),
                    answer(row.withHiddenLayer(), row.lightOn()));
        }
        System.out.println();
        System.out.printf("  Without a hidden layer: %d of %d correct, %d unsure (close to 50%%)%n", correctWithout, rows.size(), unsureWithout);
        System.out.printf("  With a hidden layer:    %d of %d correct, %d unsure (close to 50%%)%n", correctWith, rows.size(), unsureWith);
        System.out.println();
        System.out.println("  No straight line separates the two \"on\" positions from the two \"off\" ones,");
        System.out.println("  so a network without a hidden layer can't learn this rule. A hidden layer can.");
        System.out.println();
    }

    // Prints your switches and what the network with the hidden layer says about them.
    static void prediction(Switches switches, LightPrediction prediction) {
        section("4. Your switches");
        System.out.printf("  Bottom switch:  %s%n", upOrDown(switches.bottomUp()));
        System.out.printf("  Top switch:     %s%n", upOrDown(switches.topUp()));
        System.out.println();
        System.out.printf(Locale.ROOT, "  On probability:  %.1f%%%n", 100 * prediction.onProbability());
        System.out.printf(              "  Prediction:      %s%n", prediction.isOn() ? "ON" : "OFF");
        System.out.println();
        System.out.println(prediction.isOn()
                ? "  Result: The network says the light is on."
                : "  Result: The network says the light is off.");
        System.out.println();
        System.out.println("  Edit YourSwitches.java and run again to explore how the result changes.");
    }

    // "on   98.7%  right", "off   2.1%  WRONG" or "?    50.3%  unsure".
    private static String answer(float onProbability, boolean lightOn) {
        if (isUnsure(onProbability)) {
            return String.format(Locale.ROOT, "%-3s %5.1f%%  unsure", "?", 100 * onProbability);
        }
        return String.format(Locale.ROOT, "%-3s %5.1f%%  %s",
                onOrOff(onProbability >= ON_THRESHOLD), 100 * onProbability, isRight(onProbability, lightOn) ? "right" : "WRONG");
    }

    // True if the answer is too close to 50% to count as an answer.
    private static boolean isUnsure(float onProbability) {
        return Math.abs(onProbability - ON_THRESHOLD) < UNSURE_MARGIN;
    }

    // True if the answer is sure and on the right side of 50%.
    private static boolean isRight(float onProbability, boolean lightOn) {
        return !isUnsure(onProbability) && (onProbability >= ON_THRESHOLD) == lightOn;
    }

    private static String upOrDown(boolean up) {
        return up ? "up" : "down";
    }

    private static String onOrOff(boolean on) {
        return on ? "on" : "off";
    }

    // Prints a section title: "--- 1. Dataset " followed by dashes up to the full width.
    private static void section(String title) {
        String start = "--- " + title + " ";
        System.out.println(start + "-".repeat(Math.max(0, WIDTH - start.length())));
    }
}
