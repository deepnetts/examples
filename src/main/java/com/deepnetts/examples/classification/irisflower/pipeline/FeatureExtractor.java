package com.deepnetts.examples.classification.irisflower.pipeline;

import com.deepnetts.examples.classification.irisflower.dto.Flower;

import java.util.List;

// Turns a flower into the numbers the network uses.
public final class FeatureExtractor {

    // Name and order of every input. The order must match extract().
    public static final List<String> FEATURE_NAMES = List.of(
            "sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm");

    public static final int NUM_FEATURES = FEATURE_NAMES.size();

    private FeatureExtractor() {
    }

    // The four measurements as they are. They are put on a 0-1 scale later, by the scaler in IrisFlower.
    public static float[] extract(Flower flower) {
        return new float[] {
                (float) flower.sepalLengthCm(),
                (float) flower.sepalWidthCm(),
                (float) flower.petalLengthCm(),
                (float) flower.petalWidthCm()
        };
    }
}
