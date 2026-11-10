package com.deepnetts.examples.image_classification.vgg16.pipeline;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

// The names of the 1000 ImageNet categories, in the order of VGG16's outputs.
public final class ImageNetLabels {

    public static final int COUNT = 1_000;

    private ImageNetLabels() {
    }

    // labels.csv holds the names in one line, separated by commas: tench, goldfish, ... , hotdog, ...
    public static String[] load(String csvPath) throws IOException {
        Path file = Path.of(csvPath);
        if (!Files.exists(file)) {
            throw new FileNotFoundException("File not found: " + file.toAbsolutePath()
                    + " (run the example from the project root folder)");
        }
        String[] labels = Files.readString(file).trim().split("\\s*,\\s*");
        if (labels.length != COUNT) {
            throw new IOException(csvPath + ": expected " + COUNT + " labels, found " + labels.length);
        }
        return labels;
    }

    // The place of a label in the list, for example where "hotdog" is.
    public static int indexOf(String[] labels, String label) throws IOException {
        for (int i = 0; i < labels.length; i++) {
            if (labels[i].equals(label)) {
                return i;
            }
        }
        throw new IOException("ImageNet has no label " + label);
    }
}
