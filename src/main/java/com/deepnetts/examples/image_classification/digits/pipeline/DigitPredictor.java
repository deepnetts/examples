package com.deepnetts.examples.image_classification.digits.pipeline;

import com.deepnetts.examples.image_classification.digits.dto.DigitPrediction;
import com.deepnetts.examples.util.Predictor;

import deepnetts.net.ConvolutionalNetwork;
import deepnetts.tensor.Tensor3D;

import java.awt.image.BufferedImage;

// Predicts which digit an image shows.
public final class DigitPredictor implements Predictor<BufferedImage, DigitPrediction> {

    private final ConvolutionalNetwork network;
    private final DigitEncoder encoder;

    public DigitPredictor(ConvolutionalNetwork network, DigitEncoder encoder) {
        this.network = network;
        this.encoder = encoder;
    }

    @Override
    public DigitPrediction predict(BufferedImage image) {
        Tensor3D inputTensor = encoder.toTensor(image);
        float[] probabilities = network.predict(inputTensor).getValues();
        int digit = encoder.toDigit(probabilities);
        return new DigitPrediction(digit, probabilities[digit]);
    }
}
