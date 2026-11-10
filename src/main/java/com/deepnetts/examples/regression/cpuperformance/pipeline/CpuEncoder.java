package com.deepnetts.examples.regression.cpuperformance.pipeline;

import com.deepnetts.examples.regression.cpuperformance.dto.Cpu;
import com.deepnetts.examples.util.CsvEncoder;

import deepnetts.tensor.Tensor1D;

import java.util.List;

// Bridge between our records and the network, and between the CSV file and the network.
public final class CpuEncoder implements CsvEncoder<Cpu> {

    public static final int NUM_INPUTS = FeatureExtractor.NUM_FEATURES;
    public static final int NUM_OUTPUTS = 1;   // one output: the performance score

    // The expected CSV header: the six CPU fields and the value to predict (performance).
    private static final List<String> HEADER = List.of(
            "machineCycleNs", "minMemoryKb", "maxMemoryKb", "cacheMemoryKb",
            "minChannels", "maxChannels", "performance");

    // The value to predict: published relative performance score of the CPU.
    private static final int TARGET_COLUMN = HEADER.size() - 1;

    // The network learns best when numbers are small, so the value to predict is divided by
    // 1200 (a bit above the largest value in the data) before training,
    // and multiplied by it again after prediction.
    // A fixed number (instead of one calculated from the data) is simple, and works the same
    // in training and in prediction, so nothing extra has to be saved with the network.
    private static final double TARGET_MAX = 1200;

    public static float scaleTarget(double target) {
        return (float) (target / TARGET_MAX);
    }

    public static float unscaleTarget(float scaledTarget) {
        return (float) (scaledTarget * TARGET_MAX);
    }

    @Override
    public List<String> header() {
        return HEADER;
    }

    @Override
    public List<String> inputNames() {
        return FeatureExtractor.FEATURE_NAMES;
    }

    @Override
    public List<String> outputNames() {
        return List.of(HEADER.get(TARGET_COLUMN));
    }

    // The correct answer of a CSV row: the value to predict, on the network's small scale.
    @Override
    public Tensor1D toTarget(String[] columns) {
        return Tensor1D.of(new float[] {scaledTarget(columns[TARGET_COLUMN])});
    }

    // Converts a CPU into the numbers the network uses (tensors).
    @Override
    public Tensor1D toTensor(Cpu cpu) {
        CpuValidator.validate(cpu);
        return Tensor1D.of(FeatureExtractor.extract(cpu));
    }

    // Converts a CSV row into a CPU.
    // CSV columns: machineCycleNs, minMemoryKb, maxMemoryKb, cacheMemoryKb, minChannels, maxChannels.
    @Override
    public Cpu toRecord(String[] columns) {
        return new Cpu(
                Integer.parseInt(columns[0]),
                Integer.parseInt(columns[1]),
                Integer.parseInt(columns[2]),
                Integer.parseInt(columns[3]),
                Integer.parseInt(columns[4]),
                Integer.parseInt(columns[5]));
    }

    // The value to predict, put on the same small scale as the inputs (see CpuEncoder).
    private static float scaledTarget(String value) {
        double target = Double.parseDouble(value);
        if (!Double.isFinite(target) || target < 0) {
            throw new IllegalArgumentException("performance must be a number >= 0, was " + value);
        }
        return scaleTarget(target);
    }
}
