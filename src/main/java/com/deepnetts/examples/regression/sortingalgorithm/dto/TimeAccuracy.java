package com.deepnetts.examples.regression.sortingalgorithm.dto;

// How close the predicted times are to the measured ones.
public record TimeAccuracy(
        int runs,
        double typicalError,        // half of the predictions are closer than this (0.18 = 18% off)
        double withinTwiceShare,    // share of predictions at most 2 times too high or too low
        double r2) {                // share of the variation explained, on the log scale
}
