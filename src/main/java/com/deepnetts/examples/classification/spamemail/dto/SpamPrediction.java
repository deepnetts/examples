package com.deepnetts.examples.classification.spamemail.dto;

// The output of the network: spam probability and the final prediction.
public record SpamPrediction(
        float spamProbability,  // 0 to 1 (closer to 1 = more likely spam)
        boolean isSpam) {       // final prediction: spam / not spam (legitimate email)
}
