package com.deepnetts.examples.regression.autoinsurance.pipeline;

import com.deepnetts.examples.regression.autoinsurance.dto.PaymentPrediction;
import com.deepnetts.examples.regression.autoinsurance.dto.PolicyGroup;
import com.deepnetts.examples.util.Predictor;

import deepnetts.net.FeedForwardNetwork;
import deepnetts.tensor.Tensor1D;

// Predicts the expected insurance payment per insured year.
public final class PaymentPredictor implements Predictor<PolicyGroup, PaymentPrediction> {

    private final FeedForwardNetwork network;
    private final PolicyEncoder encoder;

    public PaymentPredictor(FeedForwardNetwork network, PolicyEncoder encoder) {
        this.network = network;
        this.encoder = encoder;
    }

    @Override
    public PaymentPrediction predict(PolicyGroup group) {
        Tensor1D input = encoder.toTensor(group);
        // The network answers on the small scale it was trained on: convert it back to real units.
        float scaledOutput = network.predict(input).getValues()[0];
        return new PaymentPrediction(PolicyEncoder.unscaleTarget(scaledOutput));
    }
}
