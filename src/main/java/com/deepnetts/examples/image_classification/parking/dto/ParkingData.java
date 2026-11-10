package com.deepnetts.examples.image_classification.parking.dto;

import deepnetts.data.ImageSet;

// The parking images, split by the time of day they were taken.
public record ParkingData(
        ImageSet trainingSet,          // pictures from earlier in the day
        ImageSet testSet,              // pictures from the end of the day, the network never sees them while training
        String firstTrainingCapture,   // time of the first training picture, like 08:05
        String lastTrainingCapture,
        String firstTestCapture,
        String lastTestCapture) {
}
