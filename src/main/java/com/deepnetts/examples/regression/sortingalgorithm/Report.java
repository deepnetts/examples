package com.deepnetts.examples.regression.sortingalgorithm;

import com.deepnetts.examples.regression.sortingalgorithm.dto.ArrayToSort;
import com.deepnetts.examples.regression.sortingalgorithm.dto.ChoiceResult;
import com.deepnetts.examples.regression.sortingalgorithm.dto.MeasuredRun;
import com.deepnetts.examples.regression.sortingalgorithm.dto.SortingAdvice;
import com.deepnetts.examples.regression.sortingalgorithm.dto.TimeAccuracy;
import com.deepnetts.examples.regression.sortingalgorithm.dto.TimePrediction;
import com.deepnetts.examples.regression.sortingalgorithm.enums.Algorithm;
import com.deepnetts.examples.regression.sortingalgorithm.pipeline.RunEncoder;

import java.util.DoubleSummaryStatistics;
import java.util.List;
import java.util.Locale;
import java.util.Map;

// Prints the output, so SortingAlgorithm contains only the workflow.
// The results are calculated in SortingEvaluator; this class only prints them.
final class Report {

    // Every section line is this wide, for example "--- 1. Dataset -------...".
    private static final int WIDTH = 80;

    private static final int EXAMPLES_SHOWN = 5;

    private Report() {
    }

    // Prints how many arrays and runs there are, and which algorithm was the fastest how often.
    static void dataset(int arrays, DoubleSummaryStatistics times, Map<Algorithm, Integer> fastestCounts) {

        section("1. Dataset");
        System.out.printf(Locale.ROOT, "  Total:      %4d arrays, each sorted by %d algorithms = %d runs%n",
                arrays, Algorithm.values().length, times.getCount());
        System.out.printf(Locale.ROOT, "  Time:       from %s to %s (us = microseconds, 1000 us = 1 ms)%n", time(times.getMin()), time(times.getMax()));
        System.out.printf(Locale.ROOT, "  Inputs:     %4d numbers per run (which algorithm, the array size and how sorted it is)%n", RunEncoder.NUM_INPUTS);
        System.out.println("  Fastest on how many arrays:");
        fastestCounts.entrySet().stream()
                .sorted(Map.Entry.<Algorithm, Integer>comparingByValue().reversed())
                .forEach(entry -> System.out.printf(Locale.ROOT, "    %-15s %4d%n", entry.getKey().displayName(), entry.getValue()));
        System.out.println();
    }

    // Prints how close the predicted times are to the measured ones, on the arrays the network never saw.
    static void evaluation(List<MeasuredRun> testRuns, double[] predicted, TimeAccuracy accuracy) {
        section("3. Evaluation");
        System.out.println();
        System.out.printf(Locale.ROOT, "  Results on %d runs (%d arrays) the network was never trained on:%n",
                accuracy.runs(), accuracy.runs() / Algorithm.values().length);
        System.out.println();
        int shown = Math.min(EXAMPLES_SHOWN, testRuns.size());
        System.out.printf(Locale.ROOT, "  First %d examples:%n", shown);
        System.out.printf(Locale.ROOT, "  %-15s %6s %7s   %12s   %12s%n", "Algorithm", "Size", "Sorted", "Actual", "Predicted");
        for (int i = 0; i < shown; i++) {
            MeasuredRun run = testRuns.get(i);
            System.out.printf(Locale.ROOT, "  %-15s %6d %6d%%   %12s   %12s%n", run.run().algorithm().displayName(),
                    run.run().arraySize(), run.run().sortedPercent(), time(run.timeMicroseconds()), time(predicted[i]));
        }
        System.out.println();

        System.out.printf(Locale.ROOT, "  Typical error  %.0f%%   - half of the predictions are closer than this to the measured time%n",
                100 * accuracy.typicalError());
        System.out.printf(Locale.ROOT, "  Within 2x      %.0f%%   - share of predictions at most 2 times too high or too low%n",
                100 * accuracy.withinTwiceShare());
        System.out.printf(Locale.ROOT, "  R2             %.3f - share of the variation explained, on the log scale "
                + "(1.0 = perfect, 0.0 = no better than the average)%n", accuracy.r2());
        System.out.println();
    }

    // Prints how good the network is at the real question: which algorithm to use.
    static void choice(ChoiceResult choice) {
        int arrays = choice.arrays();
        String usual = choice.usual().displayName();
        double best = choice.bestMicroseconds();

        System.out.printf(Locale.ROOT, "  The choice, on the %d test arrays:%n", arrays);
        System.out.printf(Locale.ROOT, "  Picked the fastest  %3d of %d  - the network's fastest was really the fastest%n",
                choice.networkFastest(), arrays);
        System.out.printf(Locale.ROOT, "  Baseline            %3d of %d  - always %s, the fastest most often in the training data%n",
                choice.usualFastest(), arrays, usual);
        System.out.printf(Locale.ROOT, "  Within 10%%         %3d of %d  - the network's pick was the fastest or at most 10%% slower "
                + "(always %s: %d)%n", choice.networkClose(), arrays, usual, choice.usualClose());
        System.out.println();
        System.out.printf(Locale.ROOT, "  Time to sort all %d arrays (measured):%n", arrays);
        System.out.printf(Locale.ROOT, "    the network's pick  %10s   %+4.0f%%%n",
                time(choice.networkMicroseconds()), 100 * (choice.networkMicroseconds() / best - 1));
        System.out.printf(Locale.ROOT, "    always %-12s %10s   %+4.0f%%%n",
                usual, time(choice.usualMicroseconds()), 100 * (choice.usualMicroseconds() / best - 1));
        System.out.printf(Locale.ROOT, "    always the fastest  %10s   - the best possible, known only after measuring all 9%n", time(best));
        System.out.println();
    }

    // Prints your array and the predicted time of every algorithm.
    static void prediction(ArrayToSort array, SortingAdvice advice) {
        double fastest = advice.fastestFirst().getFirst().timeMicroseconds();
        section("4. Your array");
        System.out.printf(Locale.ROOT, "  Size:     %d numbers%n", array.size());
        System.out.printf(Locale.ROOT, "  Sorted:   %d%% already in their place%n", array.sortedPercent());
        System.out.println();
        System.out.printf(Locale.ROOT, "  %-15s %12s   %s%n", "Algorithm", "Predicted", "Compared with the fastest");
        for (TimePrediction prediction : advice.fastestFirst()) {
            System.out.printf(Locale.ROOT, "  %-15s %12s   %.1f x%n", prediction.algorithm().displayName(),
                    time(prediction.timeMicroseconds()), prediction.timeMicroseconds() / fastest);
        }
        System.out.println();
        System.out.printf(Locale.ROOT, "  Result: The network expects %s to be the fastest for your array.%n",
                advice.fastestFirst().getFirst().algorithm().displayName());
        System.out.println();
        System.out.println("  The times are for the computer that made the dataset; on yours they differ, but the order is similar.");
        System.out.println("  Edit YourArray.java and run again to explore how the result changes.");
    }

    // A time in microseconds, as us or ms.
    private static String time(double microseconds) {
        return microseconds < 1000
                ? String.format(Locale.ROOT, "%.1f us", microseconds)
                : String.format(Locale.ROOT, "%.2f ms", microseconds / 1000);
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
