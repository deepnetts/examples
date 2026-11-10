package com.deepnetts.examples.classification.sonarmines.pipeline;

import com.deepnetts.examples.classification.sonarmines.dto.MinePrediction;
import com.deepnetts.examples.classification.sonarmines.dto.SonarContact;
import com.deepnetts.examples.util.Predictor;

import deepnetts.net.FeedForwardNetwork;
import deepnetts.tensor.Tensor1D;

// Predicts whether one sonar contact is a mine.
public final class MinePredictor implements Predictor<SonarContact, MinePrediction> {

    private final FeedForwardNetwork network;
    private final ContactEncoder encoder;
    private final float mineThreshold;

    // At or above this probability (0 to 1, e.g. 0.5), a contact counts as a mine.
    public MinePredictor(FeedForwardNetwork network, ContactEncoder encoder, float mineThreshold) {
        if (mineThreshold <= 0 || mineThreshold >= 1) {
            throw new IllegalArgumentException("mineThreshold must be between 0 and 1, was " + mineThreshold);
        }
        this.network = network;
        this.encoder = encoder;
        this.mineThreshold = mineThreshold;
    }

    @Override
    public MinePrediction predict(SonarContact contact) {
        Tensor1D input = encoder.toTensor(contact);
        // The network rescales the input numbers itself: the scaler was saved together with it.
        float mineProbability = network.predict(input).getValues()[0];
        return new MinePrediction(mineProbability, mineProbability >= mineThreshold);
    }
}
