package com.deepnetts.examples.image_classification.digits.dto;

// One kind of mistake: a real digit the network read as another digit, and how many times.
public record DigitMistake(
        int actual,      // the real digit
        int predicted,   // what the network said
        int count) {     // how many test images got this wrong answer
}
