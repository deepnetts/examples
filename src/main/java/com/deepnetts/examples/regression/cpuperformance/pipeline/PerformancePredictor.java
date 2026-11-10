package com.deepnetts.examples.regression.cpuperformance.pipeline;

import com.deepnetts.examples.regression.cpuperformance.dto.Cpu;
import com.deepnetts.examples.regression.cpuperformance.dto.PerformancePrediction;
import com.deepnetts.examples.util.Predictor;

import deepnetts.net.FeedForwardNetwork;
import deepnetts.tensor.Tensor1D;

// Predicts the performance score of a CPU.
public final class PerformancePredictor implements Predictor<Cpu, PerformancePrediction> {

    private final FeedForwardNetwork network;
    private final CpuEncoder encoder;

    public PerformancePredictor(FeedForwardNetwork network, CpuEncoder encoder) {
        this.network = network;
        this.encoder = encoder;
    }

    @Override
    public PerformancePrediction predict(Cpu cpu) {
        Tensor1D input = encoder.toTensor(cpu);
        // The network answers on the small scale it was trained on: convert it back to real units.
        float scaledOutput = network.predict(input).getValues()[0];
        return new PerformancePrediction(CpuEncoder.unscaleTarget(scaledOutput));
    }
}
