package com.deepnetts.examples.classification.spamemail.pipeline;

import com.deepnetts.examples.classification.spamemail.dto.Email;
import com.deepnetts.examples.classification.spamemail.dto.SpamPrediction;
import com.deepnetts.examples.util.Predictor;

import deepnetts.net.FeedForwardNetwork;
import deepnetts.tensor.Tensor1D;

// Predicts whether one email is spam.
public final class SpamPredictor implements Predictor<Email, SpamPrediction> {

    private final FeedForwardNetwork network;
    private final EmailEncoder encoder;
    private final float spamThreshold;

    // At or above this probability (0 to 1, e.g. 0.5), an email counts as spam.
    public SpamPredictor(FeedForwardNetwork network, EmailEncoder encoder, float spamThreshold) {
        if (spamThreshold <= 0 || spamThreshold >= 1) {
            throw new IllegalArgumentException("spamThreshold must be between 0 and 1, was " + spamThreshold);
        }
        this.network = network;
        this.encoder = encoder;
        this.spamThreshold = spamThreshold;
    }

    @Override
    public SpamPrediction predict(Email email) {
        Tensor1D input = encoder.toTensor(email);
        // The network rescales the input numbers itself: the scaler was saved together with it.
        float spamProbability = network.predict(input).getValues()[0];
        return new SpamPrediction(spamProbability, spamProbability >= spamThreshold);
    }
}
