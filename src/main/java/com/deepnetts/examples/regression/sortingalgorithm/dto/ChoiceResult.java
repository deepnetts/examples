package com.deepnetts.examples.regression.sortingalgorithm.dto;

import com.deepnetts.examples.regression.sortingalgorithm.enums.Algorithm;

// How good the network's choice of algorithm is on the test arrays, next to always using one algorithm.
public record ChoiceResult(
        int arrays,
        Algorithm usual,              // the baseline: the fastest most often in the training data
        int networkFastest,           // arrays where the network's pick really was the fastest
        int usualFastest,
        int networkClose,             // arrays where the pick was at most 10% slower than the fastest
        int usualClose,
        double networkMicroseconds,   // measured time to sort all test arrays with each choice
        double usualMicroseconds,
        double bestMicroseconds) {    // always the fastest: known only after measuring all 9
}
