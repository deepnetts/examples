package com.deepnetts.examples.image_classification.hotdog.pipeline;

import deepnetts.data.ExampleImage;
import deepnetts.data.ImageSet;
import deepnetts.tensor.Tensor1D;
import deepnetts.tensor.Tensor3D;
import deepnetts.util.ImageUtils;

import java.awt.image.BufferedImage;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import javax.imageio.ImageIO;

// Bridge between images and the network, and between the image folder and the network.
public final class HotDogEncoder {

    // Every photo is stretched to this size, so the network always gets the same number of pixels.
    // The photos are up to 128 pixels wide. 32 x 32 trains fast; bigger sizes (48, 64) were tried
    // and gave no better results on these 467 photos, only slower training.
    public static final int IMAGE_WIDTH = 32;
    public static final int IMAGE_HEIGHT = 32;
    public static final int IMAGE_CHANNELS = 3;   // red, green and blue
    public static final int NUM_OUTPUTS = 1;      // one output: the probability that the photo shows a hot dog

    // labels.txt holds the one label we look for (hot_dog).
    // index.txt lists the images, one per line: the path inside the folder and the label (hot_dog or negative).
    private static final String LABELS_FILE = "labels.txt";
    private static final String INDEX_FILE = "index.txt";

    // Converts a photo into the numbers the network uses (tensor).
    // These are the same steps as for the photos the network was trained on:
    // the photo is shrunk to 32 x 32 and every pixel value is put between 0 and 1.
    public Tensor3D toTensor(BufferedImage image) {
        BufferedImage scaledImage = ImageUtils.scaleImage(image, IMAGE_WIDTH, IMAGE_HEIGHT);
        return new ExampleImage(scaledImage).getInput();
    }

    // Reads the folder with images into a dataset the network can train on.
    // Every image is stretched to the size the network uses and gets a target: 1 = hot dog, 0 = anything else.
    // The images are read one by one in the order of their path, so the split into training and test images
    // is the same on every run. The error names the folder or the image.
    public ImageSet load(String folderPath) throws IOException {
        Path folder = Path.of(folderPath);
        List<String> labels = lines(folder.resolve(LABELS_FILE));
        if (labels.size() != NUM_OUTPUTS) {
            throw new IOException(LABELS_FILE + ": expected 1 label (hot_dog), found " + labels.size());
        }
        String hotDogLabel = labels.getFirst();

        ImageSet data = new ImageSet(IMAGE_WIDTH, IMAGE_HEIGHT);
        data.setColumnNames(new String[] {hotDogLabel});
        for (String line : lines(folder.resolve(INDEX_FILE)).stream().sorted().toList()) {
            String[] parts = line.split("\\s+");
            Path file = folder.resolve(parts[0]);
            String label = parts.length > 1 ? parts[1] : file.getParent().getFileName().toString();
            BufferedImage image = ImageUtils.scaleImage(read(file), IMAGE_WIDTH, IMAGE_HEIGHT);
            ExampleImage example = new ExampleImage(file.toFile(), image, label);
            example.setTargetOutput(Tensor1D.of(new float[] {label.equals(hotDogLabel) ? 1f : 0f}));
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
}
