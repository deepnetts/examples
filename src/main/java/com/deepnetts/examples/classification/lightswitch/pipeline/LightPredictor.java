package com.deepnetts.examples.classification.lightswitch.pipeline;

import com.deepnetts.examples.classification.lightswitch.dto.LightPrediction;
import com.deepnetts.examples.classification.lightswitch.dto.Switches;
import com.deepnetts.examples.util.Predictor;

import deepnetts.net.FeedForwardNetwork;

// Predicts whether the light is on for one position of the two switches.
public final class LightPredictor implements Predictor<Switches, LightPrediction> {

    // At or above this probability, the light counts as on.
    private static final float ON_THRESHOLD = 0.5f;

    private final FeedForwardNetwork network;
    private final SwitchesEncoder encoder;

    public LightPredictor(FeedForwardNetwork network, SwitchesEncoder encoder) {
        this.network = network;
        this.encoder = encoder;
    }

    @Override
    public LightPrediction predict(Switches switches) {
        float onProbability = network.predict(encoder.toTensor(switches)).getValues()[0];
        return new LightPrediction(onProbability, onProbability >= ON_THRESHOLD);
    }
}
