package com.deepnetts.examples.regression.sortingalgorithm.dto;

import com.deepnetts.examples.regression.sortingalgorithm.enums.Algorithm;

// The input of the network: one algorithm sorting one array.
public record SortRun(
        Algorithm algorithm,
        int arraySize,         // how many numbers the array has
        int sortedPercent) {   // share of the numbers already in their place, from 0 (random order) to 100 (sorted)
}
