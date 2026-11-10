package com.deepnetts.examples.regression.sortingalgorithm.dto;

import com.deepnetts.examples.regression.sortingalgorithm.enums.Algorithm;

// The output of the network for one algorithm: how long it would take to sort the array.
public record TimePrediction(
        Algorithm algorithm,
        double timeMicroseconds) {
}
