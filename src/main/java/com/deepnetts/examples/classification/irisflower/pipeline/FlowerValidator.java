package com.deepnetts.examples.classification.irisflower.pipeline;

import com.deepnetts.examples.classification.irisflower.dto.Flower;

// Checks that a flower has valid values.
public final class FlowerValidator {

    private FlowerValidator() {
    }

    // Same check, but the error says where the flower came from.
    public static void validate(Flower flower, String source) {
        try {
            validate(flower);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(source + ": " + e.getMessage(), e);
        }
    }

    // Throws an error naming the broken rule and the bad value.
    public static void validate(Flower flower) {
        if (flower == null) {
            throw new IllegalArgumentException("flower must not be null");
        }
        if (!isPositiveNumber(flower.sepalLengthCm())) {
            reject("sepalLengthCm must be a number > 0", flower.sepalLengthCm());
        }
        if (!isPositiveNumber(flower.sepalWidthCm())) {
            reject("sepalWidthCm must be a number > 0", flower.sepalWidthCm());
        }
        if (!isPositiveNumber(flower.petalLengthCm())) {
            reject("petalLengthCm must be a number > 0", flower.petalLengthCm());
        }
        if (!isPositiveNumber(flower.petalWidthCm())) {
            reject("petalWidthCm must be a number > 0", flower.petalWidthCm());
        }
    }

    // Rejects "not a number" (NaN), infinity, zero and negative sizes.
    private static boolean isPositiveNumber(double value) {
        return Double.isFinite(value) && value > 0;
    }

    // Throws an error naming the broken rule and the bad value: "<rule>, was <value>".
    private static void reject(String rule, Object actualValue) {
        throw new IllegalArgumentException(rule + ", was " + actualValue);
    }
}
