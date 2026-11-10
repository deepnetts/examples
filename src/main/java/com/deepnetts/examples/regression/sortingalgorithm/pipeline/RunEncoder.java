package com.deepnetts.examples.regression.sortingalgorithm.pipeline;

import com.deepnetts.examples.regression.sortingalgorithm.dto.MeasuredRun;
import com.deepnetts.examples.regression.sortingalgorithm.dto.SortRun;
import com.deepnetts.examples.regression.sortingalgorithm.enums.Algorithm;
import com.deepnetts.examples.util.CsvEncoder;

import deepnetts.data.DataSetItem;
import deepnetts.data.MLDataItem;
import deepnetts.data.TabularDataSet;
import deepnetts.tensor.Tensor1D;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

// Bridge between our records and the network, and between the CSV file and the network.
public final class RunEncoder implements CsvEncoder<SortRun> {

    public static final int NUM_INPUTS = FeatureExtractor.NUM_FEATURES;
    public static final int NUM_OUTPUTS = 1;   // one output: the time

    // The expected CSV header.
    private static final List<String> HEADER =
            List.of("arrayId", "algorithm", "arraySize", "sortedPercent", "timeMicroseconds");

    // The array a row belongs to: the 9 algorithms that sorted the same array share this number.
    private static final int ARRAY_COLUMN = 0;

    // The value to predict: how long the sorting took, in microseconds.
    private static final int TARGET_COLUMN = HEADER.size() - 1;

    // The times go from under 1 microsecond to over 30 000: a sorted array of 100 numbers versus bubble sort of 5000.
    // On that range, learning the time itself would mean learning only the slowest runs, so the network learns its log:
    // log10 of the time in microseconds, from -1 (0.1 microseconds) to 5 (100 000 microseconds = 0.1 s) mapped to 0 to 1.
    // An error on this scale is a share of the time (10% too high), the same for a fast and a slow run.
    private static final double LOG_MIN = -1;
    private static final double LOG_MAX = 5;

    public static float scaleTarget(double timeMicroseconds) {
        return (float) ((Math.log10(timeMicroseconds) - LOG_MIN) / (LOG_MAX - LOG_MIN));
    }

    public static double unscaleTarget(float scaledTarget) {
        return Math.pow(10, LOG_MIN + scaledTarget * (LOG_MAX - LOG_MIN));
    }

    // Converts a sort run into the numbers the network uses (tensors).
    @Override
    public Tensor1D toTensor(SortRun run) {
        RunValidator.validate(run);
        return Tensor1D.of(FeatureExtractor.extract(run));
    }

    // Converts a CSV row into a sort run.
    // CSV columns: arrayId, algorithm, arraySize, sortedPercent, timeMicroseconds (columns 1 to 3 are used here).
    @Override
    public SortRun toRecord(String[] columns) {
        return new SortRun(Algorithm.fromName(columns[1]), Integer.parseInt(columns[2]), Integer.parseInt(columns[3]));
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

    // The correct answer of a CSV row: the time, on the log scale above.
    @Override
    public Tensor1D toTarget(String[] columns) {
        return Tensor1D.of(new float[] {scaleTarget(time(columns[TARGET_COLUMN]))});
    }

    // Each CSV row becomes a measured run: the array it belongs to, the sort run (the input) and the time (the correct answer).
    // Unlike load(), it keeps the array of every run, so all 9 runs of an array can go to the same side of the split.
    // The error names the file and the line.
    public List<MeasuredRun> loadRuns(String csvPath) throws IOException {
        List<MeasuredRun> runs = new ArrayList<>();
        forEachRow(csvPath, columns -> {
            SortRun run = toRecord(columns);
            RunValidator.validate(run);
            runs.add(new MeasuredRun(Integer.parseInt(columns[ARRAY_COLUMN]), run, time(columns[TARGET_COLUMN])));
        });
        return runs;
    }

    // The runs as a dataset the network can train on: the inputs of every run and its time as the correct answer.
    public TabularDataSet<MLDataItem> toDataSet(List<MeasuredRun> runs) {
        TabularDataSet<MLDataItem> data = new TabularDataSet<>(NUM_INPUTS, NUM_OUTPUTS);
        List<String> names = new ArrayList<>(inputNames());
        names.addAll(outputNames());
        data.setColumnNames(names.toArray(String[]::new));
        for (MeasuredRun measured : runs) {
            data.add(new DataSetItem(toTensor(measured.run()), Tensor1D.of(new float[] {scaleTarget(measured.timeMicroseconds())})));
        }
        return data;
    }

    private static double time(String value) {
        double time = Double.parseDouble(value);
        if (!Double.isFinite(time) || time <= 0) {
            throw new IllegalArgumentException("timeMicroseconds must be a number above 0, was " + value);
        }
        return time;
    }
}
