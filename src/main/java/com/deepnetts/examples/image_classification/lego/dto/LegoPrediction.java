package com.deepnetts.examples.image_classification.lego.dto;

// The output of the network: how sure it is that the picture shows a Lego figure.
public record LegoPrediction(
        float legoProbability,   // probability that the picture shows a Lego figure (0.0 - 1.0)
        boolean isLego) {        // true if the probability is at or above the threshold
}
