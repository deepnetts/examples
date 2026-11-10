package com.deepnetts.examples.classification.exampass.pipeline;

import com.deepnetts.examples.classification.exampass.dto.PassPrediction;
import com.deepnetts.examples.classification.exampass.dto.Student;
import com.deepnetts.examples.util.Predictor;

import deepnetts.net.FeedForwardNetwork;

// Predicts whether one student will pass the exam.
public final class PassPredictor implements Predictor<Student, PassPrediction> {

    private final FeedForwardNetwork network;
    private final StudentEncoder encoder;
    private final float passThreshold;

    // At or above this probability (0 to 1, e.g. 0.5), the student counts as passing.
    public PassPredictor(FeedForwardNetwork network, StudentEncoder encoder, float passThreshold) {
        if (passThreshold <= 0 || passThreshold >= 1) {
            throw new IllegalArgumentException("passThreshold must be between 0 and 1, was " + passThreshold);
        }
        this.network = network;
        this.encoder = encoder;
        this.passThreshold = passThreshold;
    }

    @Override
    public PassPrediction predict(Student student) {
        float passProbability = network.predict(encoder.toTensor(student)).getValues()[0];
        return new PassPrediction(passProbability, passProbability >= passThreshold);
    }
}
