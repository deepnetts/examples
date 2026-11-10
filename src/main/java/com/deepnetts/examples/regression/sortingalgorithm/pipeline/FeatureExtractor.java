package com.deepnetts.examples.regression.sortingalgorithm.pipeline;

import com.deepnetts.examples.regression.sortingalgorithm.dto.SortRun;
import com.deepnetts.examples.regression.sortingalgorithm.enums.Algorithm;

import java.util.ArrayList;
import java.util.List;

// Turns a sort run into the numbers the network uses.
public final class FeatureExtractor {

    // Name and order of every input. The order must match extract().
    // One input per algorithm (1 for the algorithm of this run, 0 for the others), then the array size, how sorted it is
    // and how many numbers are out of place.
    public static final List<String> FEATURE_NAMES = names();

    public static final int NUM_FEATURES = FEATURE_NAMES.size();

    // The smallest and the largest array in the data.
    private static final double MIN_ARRAY_SIZE = 100;
    private static final double MAX_ARRAY_SIZE = 5000;
    private static final double MAX_SORTED_PERCENT = 100;

    private FeatureExtractor() {
    }

    public static float[] extract(SortRun run) {
        float[] features = new float[NUM_FEATURES];
        features[run.algorithm().ordinal()] = 1;
        features[Algorithm.values().length]     = arraySize(run.arraySize());
        features[Algorithm.values().length + 1] = (float) (run.sortedPercent() / MAX_SORTED_PERCENT);
        features[Algorithm.values().length + 2] = outOfPlace(run.arraySize(), run.sortedPercent());
        return features;
    }

    // How many numbers are out of their place, on a log scale from 0 (none) to 1 (all 5000).
    // The sorted percent alone hides what insertion and bubble sort react to: 99% sorted is 1 number out of place
    // in an array of 100, but 50 in an array of 5000. And 0, 1 or 10 numbers out of place differ a lot,
    // which a log shows and a percent does not.
    private static float outOfPlace(int size, int sortedPercent) {
        double count = size * (MAX_SORTED_PERCENT - sortedPercent) / MAX_SORTED_PERCENT;
        return (float) (Math.log1p(count) / Math.log1p(MAX_ARRAY_SIZE));
    }

    // The size on a log scale, from 0 (100 numbers) to 1 (5000 numbers).
    // A log, because the time grows with the size much faster than in a straight line (up to size x size),
    // and on a log scale 100 -> 200 is as big a step as 2500 -> 5000.
    private static float arraySize(int size) {
        return (float) (Math.log(size / MIN_ARRAY_SIZE) / Math.log(MAX_ARRAY_SIZE / MIN_ARRAY_SIZE));
    }

    private static List<String> names() {
        List<String> names = new ArrayList<>();
        for (Algorithm algorithm : Algorithm.values()) {
            names.add("is_" + algorithm.name().toLowerCase());
        }
        names.add("array_size");
        names.add("sorted_percent");
        names.add("out_of_place");
        return List.copyOf(names);
    }
}
