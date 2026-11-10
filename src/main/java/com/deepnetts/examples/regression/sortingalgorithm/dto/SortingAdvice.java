package com.deepnetts.examples.regression.sortingalgorithm.dto;

import java.util.List;

// The predicted time of every algorithm for one array, fastest first.
public record SortingAdvice(
        List<TimePrediction> fastestFirst) {   // the first one is the fastest
}
