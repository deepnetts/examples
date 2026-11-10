package com.deepnetts.examples.regression.sortingalgorithm.pipeline;

import com.deepnetts.examples.regression.sortingalgorithm.dto.ArrayToSort;
import com.deepnetts.examples.regression.sortingalgorithm.dto.SortRun;
import com.deepnetts.examples.regression.sortingalgorithm.enums.Algorithm;

// Checks that an array and a sort run have valid values.
public final class RunValidator {

    // Allows for rounding: 1.0000001 still counts as 1.
    private static final float TOLERANCE = 1e-6f;

    private RunValidator() {
    }

    // Same check, but the error says where the array came from.
    public static void validate(ArrayToSort array, String source) {
        try {
            if (array == null) {
                throw new IllegalArgumentException("array must not be null");
            }
            checkArray(array.size(), array.sortedPercent());
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(source + ": " + e.getMessage(), e);
        }
    }

    // Throws an error naming the broken rule and the bad value.
    public static void validate(SortRun run) {
        if (run == null) {
            throw new IllegalArgumentException("sort run must not be null");
        }
        if (run.algorithm() == null) {
            throw new IllegalArgumentException("algorithm must not be null");
        }
        checkArray(run.arraySize(), run.sortedPercent());
    }

    private static void checkArray(int size, int sortedPercent) {
        // Sorting one number takes no work, so there is nothing to predict.
        if (size < 2) {
            reject("size must be at least 2", size);
        }
        if (sortedPercent < 0 || sortedPercent > 100) {
            reject("sortedPercent must be from 0 to 100", sortedPercent);
        }
    }

    // Throws an error naming the broken rule and the bad value: "<rule>, was <value>".
    private static void reject(String rule, Object actualValue) {
        throw new IllegalArgumentException(rule + ", was " + actualValue);
    }

    // The network learned array sizes from 100 to 5000 (on a 0 to 1 scale).
    // Outside that range it can only guess, and nothing else in the output would show it.
    // Prints one warning for every such input of the array. The example still runs.
    public static void warnIfOutsideData(ArrayToSort array, String source) {
        // The algorithm doesn't matter here: only the size and the order of the array are checked.
        float[] inputs = FeatureExtractor.extract(new SortRun(Algorithm.QUICK, array.size(), array.sortedPercent()));
        for (int i = 0; i < inputs.length; i++) {
            String name = FeatureExtractor.FEATURE_NAMES.get(i);
            if (inputs[i] > 1 + TOLERANCE) {
                warn(source, name, "above the largest value in the data");
            } else if (inputs[i] < -TOLERANCE) {
                warn(source, name, "below the smallest value in the data");
            }
        }
    }

    private static void warn(String source, String input, String where) {
        System.out.println("Warning: " + source + ": " + input + " is " + where
                + ", so the network can only guess. The example still runs.");
    }
}
