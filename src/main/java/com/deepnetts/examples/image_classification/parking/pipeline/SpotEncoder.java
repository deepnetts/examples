package com.deepnetts.examples.image_classification.parking.pipeline;

import com.deepnetts.examples.image_classification.parking.dto.ParkingData;

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
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.imageio.ImageIO;

// Bridge between images and the network, and between the image folder and the network.
public final class SpotEncoder {

    // Every picture of a parking spot is shrunk to this size, so the network always gets the same number of pixels.
    // The pictures are 150 x 150; 32 x 32 is still enough to see if a car is there, and it trains in seconds.
    public static final int IMAGE_WIDTH = 32;
    public static final int IMAGE_HEIGHT = 32;
    public static final int IMAGE_CHANNELS = 3;   // red, green and blue
    public static final int NUM_OUTPUTS = 1;      // one output: the probability that the spot is taken

    // labels.txt holds the one label we look for (busy).
    // index.txt lists the images, one per line: the path inside the folder and the label (busy or negative).
    // Every file name tells when the picture was taken and which spot it shows:
    // 20150703_1705_14.jpg = 3 July 2015, 17:05, spot 14.
    private static final String LABELS_FILE = "labels.txt";
    private static final String INDEX_FILE = "index.txt";

    // Converts a picture into the numbers the network uses (tensor).
    // These are the same steps as for the pictures the network was trained on:
    // the picture is shrunk to 32 x 32 and every pixel value is put between 0 and 1.
    public Tensor3D toTensor(BufferedImage image) {
        BufferedImage scaledImage = ImageUtils.scaleImage(image, IMAGE_WIDTH, IMAGE_HEIGHT);
        return new ExampleImage(scaledImage).getInput();
    }

    // Reads the folder with parking images into a training set and a test set, split by the time of day.
    // The camera takes a picture of the whole parking lot every 5 minutes, and every picture is cut into spots.
    // A normal random split would put a spot at 10:00 into training and the same spot at 10:05 into the test:
    // almost the same picture, so the test would be too easy. Instead, the last part of the day is the test set
    // (testShare = 0.2 means the last 20% of the pictures in time).
    // Pictures 5 minutes apart are almost the same, so training keeps only every n-th one (trainingStep).
    // Every image is shrunk to the size the network uses and gets a target: 1 = busy, 0 = free.
    // The error names the folder, the line or the image.
    public ParkingData load(String folderPath, double testShare, int trainingStep) throws IOException {
        Path folder = Path.of(folderPath);
        List<String> labels = lines(folder.resolve(LABELS_FILE));
        if (labels.size() != NUM_OUTPUTS) {
            throw new IOException(LABELS_FILE + ": expected 1 label (busy), found " + labels.size());
        }
        String busyLabel = labels.getFirst();

        // Sorted by file name, so by the time the picture was taken (not by folder).
        List<String> index = lines(folder.resolve(INDEX_FILE)).stream()
                .sorted(Comparator.comparing(line -> fileName(line.split("\\s+")[0])))
                .toList();
        // Numbers the capture times in order: 0 = the first picture of the day, 1 = five minutes later...
        Map<String, Integer> timeNumber = new HashMap<>();
        for (String line : index) {
            timeNumber.putIfAbsent(captureTime(line), timeNumber.size());
        }
        int firstTestTime = (int) (timeNumber.size() * (1 - testShare));

        ImageSet trainingSet = new ImageSet(IMAGE_WIDTH, IMAGE_HEIGHT);
        ImageSet testSet = new ImageSet(IMAGE_WIDTH, IMAGE_HEIGHT);
        trainingSet.setColumnNames(new String[] {busyLabel});
        testSet.setColumnNames(new String[] {busyLabel});
        String[] times = new String[timeNumber.size()];
        timeNumber.forEach((time, number) -> times[number] = time);
        for (String line : index) {
            int number = timeNumber.get(captureTime(line));
            boolean isTest = number >= firstTestTime;
            if (!isTest && number % trainingStep != 0) {
                continue;   // too close to a picture that is already in the training set
            }
            String[] parts = line.split("\\s+");
            Path file = folder.resolve(parts[0]);
            String label = parts.length > 1 ? parts[1] : file.getParent().getFileName().toString();
            BufferedImage image = ImageUtils.scaleImage(read(file), IMAGE_WIDTH, IMAGE_HEIGHT);
            ExampleImage example = new ExampleImage(file.toFile(), image, label);
            example.setTargetOutput(Tensor1D.of(new float[] {label.equals(busyLabel) ? 1f : 0f}));
            (isTest ? testSet : trainingSet).add(example);
        }
        int lastTrainingTime = (firstTestTime - 1) / trainingStep * trainingStep;
        return new ParkingData(trainingSet, testSet, clock(times[0]), clock(times[lastTrainingTime]),
                clock(times[firstTestTime]), clock(times[times.length - 1]));
    }

    // busy/20150703_1705_14.jpg -> 20150703_1705_14.jpg
    private static String fileName(String path) {
        return Path.of(path).getFileName().toString();
    }

    // The part of the file name before the spot number: 20150703_1705_14.jpg -> 20150703_1705
    private static String captureTime(String line) throws IOException {
        String name = fileName(line.split("\\s+")[0]);
        int spotStart = name.lastIndexOf('_');
        if (spotStart < 0) {
            throw new IOException(INDEX_FILE + ": expected a name like 20150703_1705_14.jpg in line: " + line);
        }
        return name.substring(0, spotStart);
    }

    // 20150703_1705 -> 17:05
    private static String clock(String captureTime) {
        String hhmm = captureTime.substring(captureTime.lastIndexOf('_') + 1);
        return hhmm.length() == 4 ? hhmm.substring(0, 2) + ":" + hhmm.substring(2) : hhmm;
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
