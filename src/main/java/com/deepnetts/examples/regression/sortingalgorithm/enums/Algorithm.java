package com.deepnetts.examples.regression.sortingalgorithm.enums;

import java.util.Arrays;

// The sorting algorithms in the data, by the name used in the CSV file.
public enum Algorithm {

    BUBBLE("Bubble Sort"),
    SELECTION("Selection Sort"),
    INSERTION("Insertion Sort"),
    ODD_EVEN("Odd-Even Sort"),
    SHELL("Shell Sort"),
    HEAP("Heap Sort"),
    MERGE("Merge Sort"),
    QUICK("Quick Sort"),
    BUCKET("Bucket Sort");

    private final String displayName;

    Algorithm(String displayName) {
        this.displayName = displayName;
    }

    public String displayName() {
        return displayName;
    }

    public static Algorithm fromName(String name) {
        return Arrays.stream(values())
                .filter(algorithm -> algorithm.displayName.equals(name))
                .findFirst()
                .orElseThrow(() -> new IllegalArgumentException("unknown algorithm \"" + name + "\", expected one of "
                        + Arrays.stream(values()).map(Algorithm::displayName).toList()));
    }
}
