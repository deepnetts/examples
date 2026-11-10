package com.deepnetts.examples.regression.taxifare.pipeline;

import com.deepnetts.examples.regression.taxifare.dto.FarePrediction;
import com.deepnetts.examples.regression.taxifare.dto.Ride;
import com.deepnetts.examples.util.Predictor;

import deepnetts.net.FeedForwardNetwork;

// Predicts the fare of one ride.
public final class FarePredictor implements Predictor<Ride, FarePrediction> {

    private final FeedForwardNetwork network;
    private final RideEncoder encoder;

    public FarePredictor(FeedForwardNetwork network, RideEncoder encoder) {
        this.network = network;
        this.encoder = encoder;
    }

    @Override
    public FarePrediction predict(Ride ride) {
        float networkOutput = network.predict(encoder.toTensor(ride)).getValues()[0];
        return new FarePrediction(encoder.toFareEur(networkOutput));
    }
}
