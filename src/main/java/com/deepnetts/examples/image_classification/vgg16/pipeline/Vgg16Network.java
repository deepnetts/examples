package com.deepnetts.examples.image_classification.vgg16.pipeline;

import deepnetts.net.ConvolutionalNetwork;
import deepnetts.net.layers.Filters;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;

// Builds the empty VGG16 network: the same layers, in the same order, as the network that was trained in TensorFlow.
// Only then can its weights be copied in, layer by layer. One layer more or less, and the weights don't fit.
public final class Vgg16Network {

    private Vgg16Network() {
    }

    public static ConvolutionalNetwork build() {
        return ConvolutionalNetwork.builder()
                // A color image of 224 x 224 pixels.
                .addInputLayer(Vgg16Encoder.IMAGE_SIZE, Vgg16Encoder.IMAGE_SIZE, Vgg16Encoder.IMAGE_CHANNELS)
                // 5 blocks of convolutional layers (3x3 filters), each block ends with max pooling (2x2).
                // Every block halves the image and doubles the number of filters (up to 512):
                // the first blocks find edges and colors, the last ones find parts of objects (an eye, a wheel).
                .addConvolutionalLayer(64, Filters.ofSize(3))
                .addConvolutionalLayer(64, Filters.ofSize(3))
                .addMaxPoolingLayer(2, 2)
                .addConvolutionalLayer(128, Filters.ofSize(3))
                .addConvolutionalLayer(128, Filters.ofSize(3))
                .addMaxPoolingLayer(2, 2)
                .addConvolutionalLayer(256, Filters.ofSize(3))
                .addConvolutionalLayer(256, Filters.ofSize(3))
                .addConvolutionalLayer(256, Filters.ofSize(3))
                .addMaxPoolingLayer(2, 2)
                .addConvolutionalLayer(512, Filters.ofSize(3))
                .addConvolutionalLayer(512, Filters.ofSize(3))
                .addConvolutionalLayer(512, Filters.ofSize(3))
                .addMaxPoolingLayer(2, 2)
                .addConvolutionalLayer(512, Filters.ofSize(3))
                .addConvolutionalLayer(512, Filters.ofSize(3))
                .addConvolutionalLayer(512, Filters.ofSize(3))
                .addMaxPoolingLayer(2, 2)
                // 2 hidden layers of 4096 units combine the parts into whole objects.
                .addFullyConnectedLayer(4_096)
                .addFullyConnectedLayer(4_096)
                // Output: 1000 probabilities, one per ImageNet category. SOFTMAX makes them add up to 1.
                .addOutputLayer(ImageNetLabels.COUNT, ActivationType.SOFTMAX)
                // RELU in every hidden layer, like the original network.
                .hiddenActivationFunction(ActivationType.RELU)
                .lossFunction(LossType.CROSS_ENTROPY)
                .build();
    }
}
