package com.deepnetts.examples.image_classification.dukedetection.pipeline;

import com.deepnetts.examples.image_classification.dukedetection.dto.DukePrediction;
import com.deepnetts.examples.util.Predictor;

import deepnetts.net.ConvolutionalNetwork;
import deepnetts.tensor.Tensor3D;

import java.awt.image.BufferedImage;

// Predicts if an image shows Duke.
public final class DukePredictor implements Predictor<BufferedImage, DukePrediction> {

    private final ConvolutionalNetwork network;
    private final DukeEncoder encoder;
    private final float dukeThreshold;   // from this probability up, the image is called Duke

    public DukePredictor(ConvolutionalNetwork network, DukeEncoder encoder, float dukeThreshold) {
        this.network = network;
        this.encoder = encoder;
        this.dukeThreshold = dukeThreshold;
    }

    @Override
    public DukePrediction predict(BufferedImage image) {
        Tensor3D inputTensor = encoder.toTensor(image);
        float dukeProbability = network.predict(inputTensor).getValues()[0];
        boolean isDuke = dukeProbability >= dukeThreshold;
        return new DukePrediction(dukeProbability, isDuke);
    }
}
