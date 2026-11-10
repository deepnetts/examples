package com.deepnetts.examples.image_classification.hotdog.pipeline;

import com.deepnetts.examples.image_classification.hotdog.dto.HotDogPrediction;
import com.deepnetts.examples.util.Predictor;

import deepnetts.net.ConvolutionalNetwork;
import deepnetts.tensor.Tensor3D;

import java.awt.image.BufferedImage;

// Predicts if a photo shows a hot dog.
public final class HotDogPredictor implements Predictor<BufferedImage, HotDogPrediction> {

    private final ConvolutionalNetwork network;
    private final HotDogEncoder encoder;
    private final float hotDogThreshold;   // from this probability up, the photo is called a hot dog

    public HotDogPredictor(ConvolutionalNetwork network, HotDogEncoder encoder, float hotDogThreshold) {
        this.network = network;
        this.encoder = encoder;
        this.hotDogThreshold = hotDogThreshold;
    }

    @Override
    public HotDogPrediction predict(BufferedImage image) {
        Tensor3D inputTensor = encoder.toTensor(image);
        float hotDogProbability = network.predict(inputTensor).getValues()[0];
        return new HotDogPrediction(hotDogProbability, hotDogProbability >= hotDogThreshold);
    }
}
