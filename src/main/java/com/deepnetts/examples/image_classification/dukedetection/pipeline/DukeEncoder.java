package com.deepnetts.examples.image_classification.dukedetection.pipeline;

import deepnetts.data.ExampleImage;
import deepnetts.data.ImageSet;
import deepnetts.tensor.Tensor3D;
import deepnetts.util.DeepNettsException;
import deepnetts.util.ImageUtils;

import java.awt.image.BufferedImage;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;

// Bridge between images and the network, and between the image folder and the network.
public final class DukeEncoder {

    // Every image is stretched to this size, so the network always gets the same number of pixels.
    public static final int IMAGE_WIDTH = 64;
    public static final int IMAGE_HEIGHT = 64;
    public static final int IMAGE_CHANNELS = 3;   // red, green and blue
    public static final int NUM_OUTPUTS = 1;      // one output: the probability that the image shows Duke

    // labels.txt holds the one label we look for (duke).
    // index.txt lists the images, one per line: the path inside the folder and the label (duke or negative).
    // Every image is loaded as "duke" (output 1) or "negative" (anything else, output 0).
    private static final String LABELS_FILE = "labels.txt";
    private static final String INDEX_FILE = "index.txt";

    // Converts an image into the numbers the network uses (tensor).
    // These are the same steps as for the images the network was trained on:
    // the image is stretched to 64 x 64 and every pixel value is put between 0 and 1.
    public Tensor3D toTensor(BufferedImage image) {
        BufferedImage scaledImage = ImageUtils.scaleImage(image, IMAGE_WIDTH, IMAGE_HEIGHT);
        return new ExampleImage(scaledImage).getInput();
    }

    // Reads the folder with images into a dataset the network can train on.
    // The error names the folder (and the image, if one cannot be read).
    public ImageSet load(String folderPath) throws IOException {
        Path folder = Path.of(folderPath);
        Path labelsFile = folder.resolve(LABELS_FILE);
        Path indexFile = folder.resolve(INDEX_FILE);
        requireFile(labelsFile);
        requireFile(indexFile);

        try {
            ImageSet data = new ImageSet(IMAGE_WIDTH, IMAGE_HEIGHT);
            String[] labels = data.loadLabels(labelsFile.toString());
            if (labels.length != NUM_OUTPUTS) {
                throw new IOException(labelsFile.getFileName() + ": expected " + NUM_OUTPUTS
                        + " label (duke), found " + labels.length);
            }
            data.loadImages(indexFile.toString());
            // Deep Netts loads the images on several threads, so their order changes from run to run.
            // Sorting them by path gives the same split into training and test images every time.
            data.getItems().sort(Comparator.comparing(image -> image.getFile().getPath()));
            return data;
        } catch (DeepNettsException e) {
            throw new IOException(folder + ": " + e.getMessage(), e);
        }
    }

    private static void requireFile(Path file) throws FileNotFoundException {
        if (!Files.exists(file)) {
            throw new FileNotFoundException("File not found: " + file.toAbsolutePath()
                    + " (run the example from the project root folder)");
        }
    }
}
