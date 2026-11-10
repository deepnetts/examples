package com.deepnetts.examples.classification.irisflower.dto;

// The input of the network: one iris flower, measured in centimeters.
public record Flower(
        double sepalLengthCm,   // sepal = the green leaf-like part under the petals
        double sepalWidthCm,
        double petalLengthCm,   // petal = the colored part of the flower
        double petalWidthCm) {
}
