package com.deepnetts.examples.image_classification.vgg16.dto;

import java.util.List;

// The output of VGG16 for one image: its best answers, the most likely first.
public record Vgg16Prediction(
        List<LabelScore> topLabels) {
}
