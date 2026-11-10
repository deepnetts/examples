package com.deepnetts.examples.regression.sortingalgorithm.dto;

// One row of the CSV file: a sort run and how long it took.
public record MeasuredRun(
        int arrayId,              // the 9 algorithms that sorted the same array share this number
        SortRun run,
        double timeMicroseconds) {
}
