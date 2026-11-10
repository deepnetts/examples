package com.deepnetts.examples.image_classification.vgg16.dto;

// One of the 1000 ImageNet categories and how sure the network is that the image shows it.
public record LabelScore(
        String label,         // for example "hotdog" or "African_elephant"
        float probability) {  // 0.0 - 1.0
}
