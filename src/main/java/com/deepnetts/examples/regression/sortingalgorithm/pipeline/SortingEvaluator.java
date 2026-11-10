package com.deepnetts.examples.regression.sortingalgorithm.pipeline;

import com.deepnetts.examples.regression.sortingalgorithm.dto.ChoiceResult;
import com.deepnetts.examples.regression.sortingalgorithm.dto.MeasuredRun;
import com.deepnetts.examples.regression.sortingalgorithm.dto.TimeAccuracy;
import com.deepnetts.examples.regression.sortingalgorithm.enums.Algorithm;

import java.util.Arrays;
import java.util.Comparator;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

// Calculates the results that Report prints: how often each algorithm is the fastest,
// how close the predicted times are, and how good the network's choice is.
public final class SortingEvaluator {

    // A pick at most this many times slower than the fastest counts as close enough (1.1 = 10% slower).
    // Two algorithms a few percent apart are a tie: measured again, the order could swap.
    public static final double CLOSE_ENOUGH = 1.1;

    private SortingEvaluator() {
    }

    // How many arrays each algorithm was the fastest on (measured times).
    public static Map<Algorithm, Integer> fastestCounts(List<MeasuredRun> runs) {
        Map<Algorithm, Integer> counts = new EnumMap<>(Algorithm.class);
        for (Map<Algorithm, Double> times : byArray(runs, measured(runs)).values()) {
            counts.merge(fastest(times), 1, Integer::sum);
        }
        return counts;
    }

    // How close the predicted times are to the measured ones.
    // The error is a factor (how many times too high or too low), the fair error for times this different (see RunEncoder).
    public static TimeAccuracy timeAccuracy(List<MeasuredRun> runs, double[] predicted) {
        int total = runs.size();
        double[] offBy = new double[total];
        double sumSquaredError = 0;
        double sumLogActual = 0;
        for (int i = 0; i < total; i++) {
            double ratio = predicted[i] / runs.get(i).timeMicroseconds();
            offBy[i] = Math.max(ratio, 1 / ratio);
            sumSquaredError += Math.pow(Math.log10(ratio), 2);
            sumLogActual += Math.log10(runs.get(i).timeMicroseconds());
        }
        // R2 compares with always predicting the average (log) time.
        double meanLog = sumLogActual / total;
        double sumBaselineSquaredError = 0;
        for (MeasuredRun run : runs) {
            sumBaselineSquaredError += Math.pow(Math.log10(run.timeMicroseconds()) - meanLog, 2);
        }
        double[] sorted = offBy.clone();
        Arrays.sort(sorted);
        long withinTwice = Arrays.stream(offBy).filter(factor -> factor <= 2).count();
        return new TimeAccuracy(total, sorted[total / 2] - 1, (double) withinTwice / total,
                1 - sumSquaredError / sumBaselineSquaredError);
    }

    // The network's choice on the test arrays: the algorithm with the smallest predicted time.
    // Compared with always using the algorithm that was the fastest most often in the training data.
    public static ChoiceResult choice(List<MeasuredRun> trainingRuns, List<MeasuredRun> testRuns, double[] predicted) {
        Algorithm usual = fastestCounts(trainingRuns).entrySet().stream()
                .max(Map.Entry.comparingByValue()).orElseThrow().getKey();
        Map<Integer, Map<Algorithm, Double>> actual = byArray(testRuns, measured(testRuns));
        Map<Integer, Map<Algorithm, Double>> predictedByArray = byArray(testRuns, predicted);

        int networkFastest = 0;
        int usualFastest = 0;
        int networkClose = 0;
        int usualClose = 0;
        double networkTime = 0;
        double usualTime = 0;
        double bestTime = 0;
        for (Map.Entry<Integer, Map<Algorithm, Double>> entry : actual.entrySet()) {
            Map<Algorithm, Double> times = entry.getValue();
            Algorithm fastest = fastest(times);
            Algorithm pick = fastest(predictedByArray.get(entry.getKey()));
            double best = times.get(fastest);
            networkFastest += pick == fastest ? 1 : 0;
            usualFastest += usual == fastest ? 1 : 0;
            networkClose += times.get(pick) <= CLOSE_ENOUGH * best ? 1 : 0;
            usualClose += times.get(usual) <= CLOSE_ENOUGH * best ? 1 : 0;
            networkTime += times.get(pick);
            usualTime += times.get(usual);
            bestTime += best;
        }
        return new ChoiceResult(actual.size(), usual, networkFastest, usualFastest, networkClose, usualClose,
                networkTime, usualTime, bestTime);
    }

    private static double[] measured(List<MeasuredRun> runs) {
        return runs.stream().mapToDouble(MeasuredRun::timeMicroseconds).toArray();
    }

    // Runs grouped by array, with the given time of each algorithm.
    private static Map<Integer, Map<Algorithm, Double>> byArray(List<MeasuredRun> runs, double[] times) {
        Map<Integer, Map<Algorithm, Double>> arrays = new TreeMap<>();
        for (int i = 0; i < runs.size(); i++) {
            MeasuredRun run = runs.get(i);
            arrays.computeIfAbsent(run.arrayId(), id -> new EnumMap<>(Algorithm.class)).put(run.run().algorithm(), times[i]);
        }
        return arrays;
    }

    private static Algorithm fastest(Map<Algorithm, Double> times) {
        return times.entrySet().stream().min(Comparator.comparingDouble(Map.Entry::getValue)).orElseThrow().getKey();
    }
}
