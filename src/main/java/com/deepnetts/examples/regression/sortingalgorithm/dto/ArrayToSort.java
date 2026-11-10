package com.deepnetts.examples.regression.sortingalgorithm.dto;

// An array you want to sort: the network says how long each algorithm would take.
public record ArrayToSort(
        int size,              // how many numbers the array has
        int sortedPercent) {   // share of the numbers already in their place, from 0 (random order) to 100 (sorted)
}
