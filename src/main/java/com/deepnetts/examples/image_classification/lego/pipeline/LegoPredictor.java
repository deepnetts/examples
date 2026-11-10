package com.deepnetts.examples.image_classification.lego.pipeline;

import com.deepnetts.examples.image_classification.lego.dto.LegoPrediction;
import com.deepnetts.examples.util.Predictor;

import deepnetts.net.ConvolutionalNetwork;
import deepnetts.tensor.Tensor3D;

import java.awt.image.BufferedImage;

// Predicts if a picture shows a Lego figure.
public final class LegoPredictor implements Predictor<BufferedImage, LegoPrediction> {

    private final ConvolutionalNetwork network;
    private final LegoEncoder encoder;
    private final float legoThreshold;   // from this probability up, the picture is called a Lego figure

    public LegoPredictor(ConvolutionalNetwork network, LegoEncoder encoder, float legoThreshold) {
        this.network = network;
        this.encoder = encoder;
        this.legoThreshold = legoThreshold;
    }

    @Override
    public LegoPrediction predict(BufferedImage image) {
        Tensor3D inputTensor = encoder.toTensor(image);
        float legoProbability = network.predict(inputTensor).getValues()[0];
        return new LegoPrediction(legoProbability, legoProbability >= legoThreshold);
    }
}
