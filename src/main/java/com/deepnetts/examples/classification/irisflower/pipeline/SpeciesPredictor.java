package com.deepnetts.examples.classification.irisflower.pipeline;

import com.deepnetts.examples.classification.irisflower.dto.Flower;
import com.deepnetts.examples.classification.irisflower.dto.SpeciesPrediction;
import com.deepnetts.examples.util.Predictor;

import deepnetts.net.FeedForwardNetwork;
import deepnetts.tensor.Tensor1D;

// Predicts the species of one flower.
public final class SpeciesPredictor implements Predictor<Flower, SpeciesPrediction> {

    private final FeedForwardNetwork network;
    private final FlowerEncoder encoder;

    public SpeciesPredictor(FeedForwardNetwork network, FlowerEncoder encoder) {
        this.network = network;
        this.encoder = encoder;
    }

    @Override
    public SpeciesPrediction predict(Flower flower) {
        Tensor1D input = encoder.toTensor(flower);
        // The network rescales the input numbers itself: the scaler was saved together with it.
        float[] probabilities = network.predict(input).getValues();
        return new SpeciesPrediction(probabilities, encoder.toSpecies(probabilities));
    }
}
