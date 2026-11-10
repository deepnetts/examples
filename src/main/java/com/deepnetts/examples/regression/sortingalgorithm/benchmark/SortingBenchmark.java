package com.deepnetts.examples.regression.sortingalgorithm.benchmark;

import com.deepnetts.examples.regression.sortingalgorithm.enums.Algorithm;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Random;

// Makes the dataset of the SortingAlgorithm example: it sorts arrays with all 9 algorithms and measures how long it takes.
// datasets/sorting-algorithm.csv was made like this on one laptop (Intel Core i5-1135G7, Java 25).
// Run it to measure your own computer (under a minute), then point DATASET in SortingAlgorithm to the new file:
//   mvn clean compile exec:java -Psorting-benchmark
// Close other programs while it runs: anything else using the CPU makes the times noisier.
public final class SortingBenchmark {

    private static final String OUTPUT = "datasets/sorting-algorithm-my-computer.csv";
    private static final int    SEED   = 7;

    private static final int ARRAYS   = 500;
    private static final int MIN_SIZE = 100;
    private static final int MAX_SIZE = 5000;

    // Each sort is repeated until it took at least this long in total (at least MIN_RUNS times), and the median run is kept.
    // A single run of a small array takes a few microseconds, which is too short to measure on its own.
    private static final long MIN_TOTAL_NANOS = 5_000_000;
    private static final int  MIN_RUNS        = 5;
    private static final int  MAX_RUNS        = 200;

    private SortingBenchmark() {
    }

    public static void main(String[] args) {
        try {
            measure();
        } catch (IOException e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
        }
    }

    private static void measure() throws IOException {
        Random random = new Random(SEED);
        System.out.println("Warming up (Java compiles the sorting code to machine code while it runs)...");
        warmUp(random);

        List<String> lines = new ArrayList<>();
        lines.add("arrayId,algorithm,arraySize,sortedPercent,timeMicroseconds");
        List<Algorithm> algorithms = new ArrayList<>(List.of(Algorithm.values()));
        for (int arrayId = 1; arrayId <= ARRAYS; arrayId++) {
            // Sizes are spread evenly on a log scale, so there are as many small arrays as large ones.
            int size = (int) Math.round(MIN_SIZE * Math.pow((double) MAX_SIZE / MIN_SIZE, random.nextDouble()));
            int sortedPercent = random.nextInt(101);
            int[] array = partlySorted(size, sortedPercent, random);

            // A different order each time, so a slower moment of the computer does not always hit the same algorithm.
            Collections.shuffle(algorithms, random);
            for (Algorithm algorithm : algorithms) {
                double micros = medianMicros(algorithm, array);
                lines.add(String.format(Locale.ROOT, "%d,%s,%d,%d,%.2f",
                        arrayId, algorithm.displayName(), size, sortedPercent, micros));
            }
            if (arrayId % 50 == 0) {
                System.out.printf(Locale.ROOT, "  %d of %d arrays%n", arrayId, ARRAYS);
            }
        }

        Path output = Path.of(OUTPUT);
        Files.createDirectories(output.toAbsolutePath().getParent());
        Files.write(output, lines);
        System.out.println("Saved " + (lines.size() - 1) + " rows to " + output.toAbsolutePath());
    }

    // An array where sortedPercent of the values are already in their place:
    // it starts sorted, then the values at the other positions are shuffled among themselves.
    static int[] partlySorted(int size, int sortedPercent, Random random) {
        int[] array = new int[size];
        for (int i = 0; i < size; i++) {
            array[i] = random.nextInt(1_000_000);
        }
        Arrays.sort(array);
        int toShuffle = (int) Math.round(size * (100 - sortedPercent) / 100.0);
        List<Integer> positions = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            positions.add(i);
        }
        Collections.shuffle(positions, random);
        List<Integer> chosen = new ArrayList<>(positions.subList(0, toShuffle));
        List<Integer> values = new ArrayList<>();
        for (int position : chosen) {
            values.add(array[position]);
        }
        Collections.shuffle(values, random);
        for (int i = 0; i < chosen.size(); i++) {
            array[chosen.get(i)] = values.get(i);
        }
        return array;
    }

    private static double medianMicros(Algorithm algorithm, int[] array) {
        List<Long> runs = new ArrayList<>();
        long total = 0;
        while (runs.size() < MAX_RUNS && (runs.size() < MIN_RUNS || total < MIN_TOTAL_NANOS)) {
            int[] copy = array.clone();   // copied before the clock starts, so only the sorting is measured
            long start = System.nanoTime();
            Sorts.sort(algorithm, copy);
            long nanos = System.nanoTime() - start;
            checkSorted(algorithm, copy);
            runs.add(nanos);
            total += nanos;
        }
        Collections.sort(runs);
        return runs.get(runs.size() / 2) / 1000.0;
    }

    private static void warmUp(Random random) {
        for (Algorithm algorithm : Algorithm.values()) {
            for (int size : new int[] {MIN_SIZE, 1000, MAX_SIZE}) {
                for (int i = 0; i < 20; i++) {
                    int[] array = partlySorted(size, random.nextInt(101), random);
                    Sorts.sort(algorithm, array);
                    checkSorted(algorithm, array);
                }
            }
        }
    }

    private static void checkSorted(Algorithm algorithm, int[] array) {
        for (int i = 1; i < array.length; i++) {
            if (array[i - 1] > array[i]) {
                throw new IllegalStateException(algorithm.displayName() + " did not sort the array");
            }
        }
    }
}
