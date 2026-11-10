package com.deepnetts.examples.image_classification.digits.dto;

// The output of the network for one image: which digit it sees and how sure it is.
public record DigitPrediction(
        int digit,            // the digit with the highest probability (0 - 9)
        float probability) {  // the network's probability for that digit (0.0 - 1.0)
}
