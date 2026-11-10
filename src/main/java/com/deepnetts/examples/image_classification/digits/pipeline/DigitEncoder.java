package com.deepnetts.examples.image_classification.digits.pipeline;

import deepnetts.data.ExampleImage;
import deepnetts.data.ImageSet;
import deepnetts.tensor.Tensor1D;
import deepnetts.tensor.Tensor3D;
import deepnetts.util.ImageUtils;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import javax.imageio.ImageIO;

// Bridge between images and the network, in both directions, and between the image folder and the network.
public final class DigitEncoder {

    // The digits are small gray pictures: 28 x 28 pixels, one value per pixel (how bright it is).
    public static final int IMAGE_WIDTH = 28;
    public static final int IMAGE_HEIGHT = 28;
    public static final int IMAGE_CHANNELS = 1;   // gray: one brightness value instead of red, green and blue
    public static final int NUM_OUTPUTS = 10;     // one output per digit, 0 to 9

    // labels.txt holds the 10 labels, 0 to 9, in the order of the network's outputs.
    // index.txt lists the images, one per line: the path inside the folder and the label (the digit).
    private static final String LABELS_FILE = "labels.txt";
    private static final String INDEX_FILE = "index.txt";

    // Converts an image into the numbers the network uses (tensor).
    // These are the same steps as for the images the network was trained on.
    public Tensor3D toTensor(BufferedImage image) {
        return new ExampleImage(toGray(image), null, IMAGE_CHANNELS).getInput();
    }

    // Shrinks the image to 28 x 28 and turns it gray: each pixel becomes the average of its red, green and blue.
    // The training images are already like this; your own image may be bigger or in color.
    public BufferedImage toGray(BufferedImage image) {
        BufferedImage scaled = image.getWidth() == IMAGE_WIDTH && image.getHeight() == IMAGE_HEIGHT
                ? image
                : ImageUtils.scaleImage(image, IMAGE_WIDTH, IMAGE_HEIGHT);
        BufferedImage gray = new BufferedImage(IMAGE_WIDTH, IMAGE_HEIGHT, BufferedImage.TYPE_BYTE_GRAY);
        for (int y = 0; y < IMAGE_HEIGHT; y++) {
            for (int x = 0; x < IMAGE_WIDTH; x++) {
                int rgb = scaled.getRGB(x, y);
                int brightness = (((rgb >> 16) & 0xFF) + ((rgb >> 8) & 0xFF) + (rgb & 0xFF)) / 3;
                gray.getRaster().setSample(x, y, 0, brightness);
            }
        }
        return gray;
    }

    // The network gives 10 probabilities, one per digit. The answer is the digit with the highest one.
    public int toDigit(float[] outputs) {
        int best = 0;
        for (int digit = 1; digit < outputs.length; digit++) {
            if (outputs[digit] > outputs[best]) {
                best = digit;
            }
        }
        return best;
    }

    // Reads the folder with digit images into a dataset the network can train on.
    // Every image is turned gray and gets a target: 10 numbers, a 1 at the place of its digit and 0 elsewhere.
    // The images are read one by one in the order of their path, so the split into training and test images
    // is the same on every run. The error names the folder, the line or the image.
    public ImageSet load(String folderPath) throws IOException {
        Path folder = Path.of(folderPath);
        List<String> labels = lines(folder.resolve(LABELS_FILE));
        if (labels.size() != NUM_OUTPUTS) {
            throw new IOException(LABELS_FILE + ": expected " + NUM_OUTPUTS + " labels (0 to 9), found " + labels.size());
        }

        ImageSet data = new ImageSet(IMAGE_WIDTH, IMAGE_HEIGHT);
        data.setColumnNames(labels.toArray(String[]::new));
        for (String line : lines(folder.resolve(INDEX_FILE)).stream().sorted().toList()) {
            String[] parts = line.split("\\s+");
            Path file = folder.resolve(parts[0]);
            String label = parts.length > 1 ? parts[1] : file.getParent().getFileName().toString();
            int digit = labels.indexOf(label);
            if (digit < 0) {
                throw new IOException(INDEX_FILE + ": unknown label " + label + " in line: " + line);
            }
            float[] target = new float[NUM_OUTPUTS];
            target[digit] = 1f;
            ExampleImage example = new DigitImage(file.toFile(), toGray(read(file)), label);
            example.setTargetOutput(Tensor1D.of(target));
            data.add(example);
        }
        return data;
    }

    private static List<String> lines(Path file) throws IOException {
        if (!Files.exists(file)) {
            throw new FileNotFoundException("File not found: " + file.toAbsolutePath()
                    + " (run the example from the project root folder)");
        }
        return Files.readAllLines(file).stream().map(String::trim).filter(line -> !line.isEmpty()).toList();
    }

    private static BufferedImage read(Path file) throws IOException {
        if (!Files.exists(file)) {
            throw new FileNotFoundException("Image not found: " + file.toAbsolutePath());
        }
        BufferedImage image = ImageIO.read(file.toFile());
        if (image == null) {
            throw new IOException(file + " is not an image file (use jpg or png)");
        }
        return image;
    }

    // A gray image that remembers its file, so the report can show which image it is.
    // (Deep Netts keeps the file only for color images.)
    private static final class DigitImage extends ExampleImage {

        private final File file;

        DigitImage(File file, BufferedImage grayImage, String label) {
            super(grayImage, label, IMAGE_CHANNELS);
            this.file = file;
        }

        @Override
        public File getFile() {
            return file;
        }
    }
}
