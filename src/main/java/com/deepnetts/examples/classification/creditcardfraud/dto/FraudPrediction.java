package com.deepnetts.examples.classification.creditcardfraud.dto;

// The output of the network: fraud probability and the final prediction.
public record FraudPrediction(
        float fraudProbability,  // 0 to 1 (closer to 1 = more likely fraud). A score, not the real chance: see the README
        boolean isFraud) {       // final prediction: fraud / legitimate
}
