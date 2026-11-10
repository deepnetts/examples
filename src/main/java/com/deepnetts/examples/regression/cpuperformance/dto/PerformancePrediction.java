package com.deepnetts.examples.regression.cpuperformance.dto;

// The output of the network: the performance score of a CPU.
public record PerformancePrediction(
        float performance) {   // published relative performance score (higher = faster CPU)
}
