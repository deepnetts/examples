package com.deepnetts.examples.regression.bostonhousing.pipeline;

import com.deepnetts.examples.regression.bostonhousing.dto.HomeValuePrediction;
import com.deepnetts.examples.regression.bostonhousing.dto.Neighborhood;
import com.deepnetts.examples.util.Predictor;

import deepnetts.net.FeedForwardNetwork;
import deepnetts.tensor.Tensor1D;

// Predicts the typical home value in a neighborhood.
public final class HomeValuePredictor implements Predictor<Neighborhood, HomeValuePrediction> {

    private final FeedForwardNetwork network;
    private final NeighborhoodEncoder encoder;

    public HomeValuePredictor(FeedForwardNetwork network, NeighborhoodEncoder encoder) {
        this.network = network;
        this.encoder = encoder;
    }

    @Override
    public HomeValuePrediction predict(Neighborhood neighborhood) {
        Tensor1D input = encoder.toTensor(neighborhood);
        // The network answers on the small scale it was trained on: convert it back to real units.
        float scaledOutput = network.predict(input).getValues()[0];
        return new HomeValuePrediction(NeighborhoodEncoder.unscaleTarget(scaledOutput));
    }
}
