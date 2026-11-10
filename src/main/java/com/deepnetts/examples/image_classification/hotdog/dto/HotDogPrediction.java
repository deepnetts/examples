package com.deepnetts.examples.image_classification.hotdog.dto;

// The output of the network: how sure it is that the photo shows a hot dog.
public record HotDogPrediction(
        float hotDogProbability,   // probability that the photo shows a hot dog (0.0 - 1.0)
        boolean isHotDog) {        // true if the probability is at or above the threshold
}
