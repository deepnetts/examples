package com.deepnetts.examples.image_classification.vgg16;

import com.deepnetts.examples.image_classification.vgg16.dto.LabelScore;
import com.deepnetts.examples.image_classification.vgg16.dto.Vgg16Prediction;
import com.deepnetts.examples.image_classification.vgg16.pipeline.Vgg16Encoder;

import java.nio.file.Path;
import java.util.Locale;

// Prints the output, so Vgg16Import contains only the workflow.
final class Report {

    // Every section line is this wide, for example "--- 1. Dataset -------...".
    private static final int WIDTH = 80;

    private Report() {
    }

    // Prints what was loaded: the network, its weights and the labels.
    static void model(Path weightsFile, int layerCount, int labelCount) {
        section("1. Model");
        System.out.printf(Locale.ROOT, "  Network:  VGG16, %d layers, input %d x %d pixels, %d colors%n",
                layerCount, Vgg16Encoder.IMAGE_SIZE, Vgg16Encoder.IMAGE_SIZE, Vgg16Encoder.IMAGE_CHANNELS);
        System.out.printf(Locale.ROOT, "  Weights:  %s (trained in TensorFlow on ImageNet)%n", weightsFile);
        System.out.printf(Locale.ROOT, "  Labels:   %d ImageNet categories%n", labelCount);
        System.out.println("  Training: none, the weights are used as they are");
        System.out.println();
    }

    // Prints VGG16's best answers for your image.
    static void prediction(Path imagePath, Vgg16Prediction prediction) {
        section("2. Your image");
        System.out.printf(Locale.ROOT, "  Image: %s%n", imagePath);
        System.out.println();
        System.out.println("  VGG16's best answers:");
        for (LabelScore score : prediction.topLabels()) {
            System.out.printf(Locale.ROOT, "    %-28s %5.1f %%%n", score.label(), score.probability() * 100);
        }
        System.out.println();
        System.out.println("  Edit YourImage.java and run again to explore how the result changes.");
    }

    // A share of a whole as a percent, or "n/a" when there is nothing to divide by.
    private static String percent(int part, int whole) {
        if (whole == 0) {
            return "   n/a";
        }
        return String.format(Locale.ROOT, "%5.1f %%", 100.0 * part / whole);
    }

    // Prints when the training starts.
    static void trainingStarted() {
        section("2. Training");
        System.out.println("  Training the network...");
    }

    // Prints how many epochs the training took.
    static void trainingFinished(int epochs) {
        System.out.printf("  Stopped after %d epochs%n", epochs);
        System.out.println();
    }

    // Prints a section title: "--- 1. Dataset " followed by dashes up to the full width.
    private static void section(String title) {
        String start = "--- " + title + " ";
        System.out.println(start + "-".repeat(Math.max(0, WIDTH - start.length())));
    }
}
