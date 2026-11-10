package com.deepnetts.examples.image_classification.parking.dto;

// The output of the network: how sure it is that a car is parked in the spot.
public record SpotPrediction(
        float busyProbability,   // probability that the spot is taken (0.0 - 1.0)
        boolean isBusy) {        // true if the probability is at or above the threshold
}
