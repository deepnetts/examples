package com.deepnetts.examples.classification.creditcardfraud.pipeline;

import com.deepnetts.examples.classification.creditcardfraud.dto.FraudPrediction;
import com.deepnetts.examples.classification.creditcardfraud.dto.Transaction;
import com.deepnetts.examples.util.Predictor;

import deepnetts.net.FeedForwardNetwork;
import deepnetts.tensor.Tensor1D;

// Predicts whether one transaction is fraud.
public final class FraudPredictor implements Predictor<Transaction, FraudPrediction> {

    private final FeedForwardNetwork network;
    private final TransactionEncoder encoder;
    private final float fraudThreshold;

    // At or above this probability (0 to 1, e.g. 0.5), a transaction counts as fraud.
    public FraudPredictor(FeedForwardNetwork network, TransactionEncoder encoder, float fraudThreshold) {
        if (fraudThreshold <= 0 || fraudThreshold >= 1) {
            throw new IllegalArgumentException("fraudThreshold must be between 0 and 1, was " + fraudThreshold);
        }
        this.network = network;
        this.encoder = encoder;
        this.fraudThreshold = fraudThreshold;
    }

    @Override
    public FraudPrediction predict(Transaction transaction) {
        Tensor1D input = encoder.toTensor(transaction);
        // The network puts the numbers on the 0-1 scale itself: the scale was saved together with it.
        float fraudProbability = network.predict(input).getValues()[0];
        return new FraudPrediction(fraudProbability, fraudProbability >= fraudThreshold);
    }
}
