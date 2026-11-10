package com.deepnetts.examples.image_classification.vgg16.pipeline;

import deepnetts.net.ConvolutionalNetwork;
import deepnetts.net.layers.AbstractLayer;
import deepnetts.net.layers.ConvolutionalLayer;
import deepnetts.net.layers.FullyConnectedLayer;
import deepnetts.net.layers.OutputLayer;
import deepnetts.tensor.Shape;
import deepnetts.tensor.Tensor1D;
import deepnetts.tensor.Tensor2D;
import deepnetts.tensor.Tensor4D;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

// Copies the weights exported from TensorFlow into the VGG16 network built by Vgg16Network.
//
// The file is text, about 2 GB. For every layer with weights it has:
//   weights_shape:3, 3, 3, 64          the shape in TensorFlow (for a filter: height, width, input channels, filters)
//   weights:                           then the weights, comma after comma, over many lines
//   biases:0.73, 0.09, ...             then the biases, in one line
// Deep Netts has TensorflowUtils.importWeights(network, file) for this. It keeps a whole layer as one piece of text
// before it turns it into numbers, and the biggest VGG16 layer has 103 million weights: that needs about 8 GB of memory.
// This reader turns the text into numbers as it reads it, and puts them into the same Deep Netts tensors in the same
// order, so about 4 GB is enough.
// Deep Netts' layers are generic types that this code uses without their type argument, so the compiler is told to allow it.
@SuppressWarnings({"rawtypes", "unchecked"})
public final class Vgg16WeightsReader {

    private Vgg16WeightsReader() {
    }

    public static void read(ConvolutionalNetwork network, Path weightsFile) throws IOException {
        // The layers that have weights, in order: 13 convolutional, 2 fully connected, 1 output.
        // (Input, pooling and TensorFlow's flatten layer have none.)
        List<AbstractLayer> layers = new ArrayList<>();
        for (AbstractLayer layer : network.getLayers()) {
            if (layer instanceof ConvolutionalLayer || layer instanceof FullyConnectedLayer || layer instanceof OutputLayer) {
                layers.add(layer);
            }
        }

        int next = 0;
        try (BufferedReader reader = Files.newBufferedReader(weightsFile, StandardCharsets.UTF_8)) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (!line.startsWith("weights_shape:")) {
                    continue;
                }
                if (next == layers.size()) {
                    throw new IOException(weightsFile + ": more layers with weights than VGG16 has (" + layers.size() + ")");
                }
                AbstractLayer layer = layers.get(next++);
                int[] shape = numbers(line.substring(line.indexOf(':') + 1));
                if (!"weights:".equals(reader.readLine())) {
                    throw new IOException(weightsFile + ": expected \"weights:\" after " + line);
                }
                if (layer instanceof ConvolutionalLayer convolutional) {
                    // TensorFlow: height, width, input channels, filters. Deep Netts: filters, input channels, height, width.
                    Tensor4D filters = new Tensor4D(shape[3], shape[2], shape[0], shape[1]);
                    ValueReader values = new ValueReader(reader);
                    for (int filter = 0; filter < shape[3]; filter++) {
                        for (int channel = 0; channel < shape[2]; channel++) {
                            for (int row = 0; row < shape[0]; row++) {
                                for (int col = 0; col < shape[1]; col++) {
                                    filters.set(values.next(), filter, channel, row, col);
                                }
                            }
                        }
                    }
                    convolutional.setFilters(filters);
                    convolutional.createFilterCache();
                    float[] biases = biases(values.rest(), shape[3]);
                    layer.setBiases(new Tensor1D(Shape.of(1, 1, biases.length), biases));
                } else {
                    // TensorFlow: inputs, outputs. Deep Netts keeps it the other way round: one row per output.
                    Tensor2D weights = new Tensor2D(shape[1], shape[0]);
                    ValueReader values = new ValueReader(reader);
                    for (int input = 0; input < shape[0]; input++) {
                        for (int output = 0; output < shape[1]; output++) {
                            weights.set(values.next(), output, input);
                        }
                    }
                    weights.createRowsCache();
                    layer.setWeights(weights);
                    layer.setBiases(new Tensor1D(biases(values.rest(), shape[1])));
                }
            }
        }
        if (next != layers.size()) {
            throw new IOException(weightsFile + ": found weights for " + next + " layers, VGG16 has " + layers.size());
        }
    }

    // "3, 3, 3, 64" -> {3, 3, 3, 64}
    private static int[] numbers(String text) {
        String[] parts = text.split(",");
        int[] values = new int[parts.length];
        for (int i = 0; i < parts.length; i++) {
            values[i] = Integer.parseInt(parts[i].trim());
        }
        return values;
    }

    // "biases:0.73, 0.09, ..." -> the numbers
    private static float[] biases(String line, int expected) throws IOException {
        if (line == null || !line.startsWith("biases:")) {
            throw new IOException("expected \"biases:\" after the weights, found: " + (line == null ? "the end of the file" : line));
        }
        String[] parts = line.substring(line.indexOf(':') + 1).split(",");
        if (parts.length != expected) {
            throw new IOException("expected " + expected + " biases, found " + parts.length);
        }
        float[] values = new float[parts.length];
        for (int i = 0; i < parts.length; i++) {
            values[i] = Float.parseFloat(parts[i].trim());
        }
        return values;
    }

    // Reads the weights one number at a time. A number ends at a comma; the lines are joined without a separator.
    private static final class ValueReader {

        private final BufferedReader reader;
        private final StringBuilder number = new StringBuilder(16);
        private String line = "";
        private int position;

        ValueReader(BufferedReader reader) {
            this.reader = reader;
        }

        float next() throws IOException {
            number.setLength(0);
            while (true) {
                while (position == line.length()) {   // also skips empty lines
                    line = reader.readLine();
                    position = 0;
                    if (line == null) {
                        throw new IOException("the weights file ends in the middle of a layer");
                    }
                    if (line.startsWith("biases:")) {
                        throw new IOException("a layer has fewer weights than its shape says");
                    }
                }
                char c = line.charAt(position++);
                if (c == ',') {
                    return Float.parseFloat(number.toString().trim());
                }
                number.append(c);
            }
        }

        // The line after the weights: it starts with "biases:". Skips what is left of the last weights line.
        String rest() throws IOException {
            String biasLine;
            do {
                biasLine = reader.readLine();
            } while (biasLine != null && biasLine.isBlank());
            return biasLine;
        }
    }
}
