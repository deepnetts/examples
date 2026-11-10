package com.deepnetts.examples.image_classification.vgg16;

import com.deepnetts.examples.image_classification.vgg16.dto.Vgg16Prediction;
import com.deepnetts.examples.image_classification.vgg16.pipeline.ImageNetLabels;
import com.deepnetts.examples.image_classification.vgg16.pipeline.Vgg16Encoder;
import com.deepnetts.examples.image_classification.vgg16.pipeline.Vgg16Network;
import com.deepnetts.examples.image_classification.vgg16.pipeline.Vgg16Predictor;
import com.deepnetts.examples.image_classification.vgg16.pipeline.Vgg16Weights;
import com.deepnetts.examples.image_classification.vgg16.pipeline.Vgg16WeightsReader;

import deepnetts.core.DeepNetts;
import deepnetts.net.ConvolutionalNetwork;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;

// VGG16 import example: uses a network that was trained somewhere else.
// VGG16 is a well-known convolutional network (University of Oxford, 2014), trained in TensorFlow on ImageNet:
// 1.2 million photos of 1000 everyday things. That took days on graphics cards. Here Deep Netts builds the same
// layers, copies the trained weights in and runs it in pure Java, without any training.
// It needs about 4 GB of free memory and a few minutes: run it with MAVEN_OPTS="-Xmx4g" (see README.md).
public class Vgg16Import {

    private static final Path   WEIGHTS_FOLDER = Path.of("models", "cache");
    private static final String LABELS         = "datasets/ImageNet/labels.csv";
    private static final int    ANSWERS_SHOWN  = 5;

    // VGG16 needs about 4 GB of memory. Java gets less by default and then fails in the middle of building the network.
    private static final long   MIN_MEMORY_BYTES = 3584L * 1024 * 1024;

    // Hides the detailed log.
    private static final Logger DEEPNETTS_LOG = Logger.getLogger("deepnetts");

    public static void main(String[] args) {
        // Prints a clear error (missing file, download failed, invalid YourImage) and stops.
        try {
            run();
        } catch (IOException | IllegalArgumentException | IllegalStateException e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
        }
    }

    // Stops with a clear message when Java may use too little memory, instead of an OutOfMemoryError later.
    static void requireMemory(long maxMemoryBytes) {
        if (maxMemoryBytes < MIN_MEMORY_BYTES) {
            throw new IllegalStateException("VGG16 needs about 4 GB of memory, but this Java may use only "
                    + maxMemoryBytes / (1024 * 1024) + " MB. Start it with MAVEN_OPTS=-Xmx4g (Maven) or the VM option -Xmx4g (IDE): see README.md.");
        }
    }

    static void run() throws IOException {

        requireMemory(Runtime.getRuntime().maxMemory());

        // VGG16 does about 15 billion multiplications per image, so it uses all processor cores.
        // There is no training here, so the result is the same on every run anyway.
        DeepNetts.getInstance().setMaxThreads(Runtime.getRuntime().availableProcessors());
        // Comment out this line to see the detailed log.
        DEEPNETTS_LOG.setLevel(Level.WARNING);

        // Reads YourImage first.
        Path imagePath = YourImage.path();
        BufferedImage image = readImage(imagePath, "YourImage");

        Vgg16Encoder encoder = new Vgg16Encoder();

        // ── 1. Model ─────────────────────────────
        // Builds the empty network: the same layers as in TensorFlow, in the same order.
        ConvolutionalNetwork vgg16 = Vgg16Network.build();
        // Gets the trained weights (downloads them the first time, about 600 MB) and copies them into the network.
        Path weightsFile = Vgg16Weights.obtain(WEIGHTS_FOLDER);
        Vgg16WeightsReader.read(vgg16, weightsFile);
        // The names of the 1000 categories, in the order of the network's outputs.
        String[] labels = ImageNetLabels.load(LABELS);
        Report.model(weightsFile, vgg16.getLayers().size(), labels.length);
        // There is nothing to save: the weights file already is the trained network.

        // ── 2. Predict your image ────────────────
        Vgg16Predictor predictor = new Vgg16Predictor(vgg16, encoder, labels, ANSWERS_SHOWN);
        Vgg16Prediction prediction = predictor.predict(image);
        Report.prediction(imagePath, prediction);
    }

    private static BufferedImage readImage(Path imagePath, String source) throws IOException {
        if (!imagePath.toFile().exists()) {
            throw new java.io.FileNotFoundException(source + ": image not found: " + imagePath.toAbsolutePath()
                    + " (run the example from the project root folder)");
        }
        BufferedImage image = ImageIO.read(imagePath.toFile());
        if (image == null) {
            throw new IOException(source + ": " + imagePath.getFileName() + " is not an image file (use jpg or png)");
        }
        return image;
    }
}
