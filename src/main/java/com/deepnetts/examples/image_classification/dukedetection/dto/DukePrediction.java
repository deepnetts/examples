package com.deepnetts.examples.image_classification.dukedetection.dto;

// The output of the network: how sure it is that the image shows Duke.
public record DukePrediction(
        float dukeProbability,   // probability that the image shows Duke (0.0 - 1.0)
        boolean isDuke) {        // true if the probability is above the threshold
}
