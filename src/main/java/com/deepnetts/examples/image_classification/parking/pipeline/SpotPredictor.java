package com.deepnetts.examples.image_classification.parking.pipeline;

import com.deepnetts.examples.image_classification.parking.dto.SpotPrediction;
import com.deepnetts.examples.util.Predictor;

import deepnetts.net.ConvolutionalNetwork;
import deepnetts.tensor.Tensor3D;

import java.awt.image.BufferedImage;

// Predicts if a parking spot is taken.
public final class SpotPredictor implements Predictor<BufferedImage, SpotPrediction> {

    private final ConvolutionalNetwork network;
    private final SpotEncoder encoder;
    private final float busyThreshold;   // from this probability up, the spot is called busy

    public SpotPredictor(ConvolutionalNetwork network, SpotEncoder encoder, float busyThreshold) {
        this.network = network;
        this.encoder = encoder;
        this.busyThreshold = busyThreshold;
    }

    @Override
    public SpotPrediction predict(BufferedImage image) {
        Tensor3D inputTensor = encoder.toTensor(image);
        float busyProbability = network.predict(inputTensor).getValues()[0];
        return new SpotPrediction(busyProbability, busyProbability >= busyThreshold);
    }
}
