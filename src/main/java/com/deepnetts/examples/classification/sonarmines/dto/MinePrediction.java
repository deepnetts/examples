package com.deepnetts.examples.classification.sonarmines.dto;

// The output of the network: mine probability and the final prediction.
public record MinePrediction(
        float mineProbability,  // 0 to 1 (closer to 1 = more likely a mine)
        boolean isMine) {       // final prediction: mine / not a mine (harmless object, e.g. a rock)
}
