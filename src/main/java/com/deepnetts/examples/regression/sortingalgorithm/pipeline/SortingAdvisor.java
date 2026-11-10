package com.deepnetts.examples.regression.sortingalgorithm.pipeline;

import com.deepnetts.examples.regression.sortingalgorithm.dto.ArrayToSort;
import com.deepnetts.examples.regression.sortingalgorithm.dto.SortRun;
import com.deepnetts.examples.regression.sortingalgorithm.dto.SortingAdvice;
import com.deepnetts.examples.regression.sortingalgorithm.dto.TimePrediction;
import com.deepnetts.examples.regression.sortingalgorithm.enums.Algorithm;
import com.deepnetts.examples.util.Predictor;

import deepnetts.net.FeedForwardNetwork;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

// Predicts how long every algorithm would take to sort an array, and puts the fastest first.
// The network predicts one time; asking it once per algorithm turns that into a choice.
public final class SortingAdvisor implements Predictor<ArrayToSort, SortingAdvice> {

    private final FeedForwardNetwork network;
    private final RunEncoder encoder;

    public SortingAdvisor(FeedForwardNetwork network, RunEncoder encoder) {
        this.network = network;
        this.encoder = encoder;
    }

    @Override
    public SortingAdvice predict(ArrayToSort array) {
        List<TimePrediction> predictions = new ArrayList<>();
        for (Algorithm algorithm : Algorithm.values()) {
            predictions.add(new TimePrediction(algorithm, time(new SortRun(algorithm, array.size(), array.sortedPercent()))));
        }
        predictions.sort(Comparator.comparingDouble(TimePrediction::timeMicroseconds));
        return new SortingAdvice(List.copyOf(predictions));
    }

    // The predicted time in microseconds. The network answers on the log scale it was trained on (see RunEncoder).
    public double time(SortRun run) {
        float scaledOutput = network.predict(encoder.toTensor(run)).getValues()[0];
        return RunEncoder.unscaleTarget(scaledOutput);
    }
}
