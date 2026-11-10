package com.deepnetts.examples.image_classification.vgg16.pipeline;

import com.deepnetts.examples.image_classification.vgg16.dto.LabelScore;
import com.deepnetts.examples.image_classification.vgg16.dto.Vgg16Prediction;
import com.deepnetts.examples.util.Predictor;

import deepnetts.net.ConvolutionalNetwork;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

// Asks VGG16 what an image shows and keeps its best answers.
public final class Vgg16Predictor implements Predictor<BufferedImage, Vgg16Prediction> {

    private final ConvolutionalNetwork network;
    private final Vgg16Encoder encoder;
    private final String[] labels;
    private final int answersKept;

    public Vgg16Predictor(ConvolutionalNetwork network, Vgg16Encoder encoder, String[] labels, int answersKept) {
        this.network = network;
        this.encoder = encoder;
        this.labels = labels;
        this.answersKept = answersKept;
    }

    @Override
    public Vgg16Prediction predict(BufferedImage image) {
        float[] probabilities = network.predict(encoder.toTensor(image)).getValues();
        return new Vgg16Prediction(best(probabilities));
    }

    // The labels with the highest probabilities, the highest first.
    private List<LabelScore> best(float[] probabilities) {
        List<LabelScore> best = new ArrayList<>();
        boolean[] taken = new boolean[probabilities.length];
        for (int n = 0; n < Math.min(answersKept, probabilities.length); n++) {
            int top = -1;
            for (int i = 0; i < probabilities.length; i++) {
                if (!taken[i] && (top < 0 || probabilities[i] > probabilities[top])) {
                    top = i;
                }
            }
            taken[top] = true;
            best.add(new LabelScore(labels[top], probabilities[top]));
        }
        return best;
    }
}
