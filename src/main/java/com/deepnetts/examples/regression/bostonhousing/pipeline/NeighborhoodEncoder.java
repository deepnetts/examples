package com.deepnetts.examples.regression.bostonhousing.pipeline;

import com.deepnetts.examples.regression.bostonhousing.dto.Neighborhood;
import com.deepnetts.examples.util.CsvEncoder;

import deepnetts.tensor.Tensor1D;

import java.util.List;

// Bridge between our records and the network, and between the CSV file and the network.
public final class NeighborhoodEncoder implements CsvEncoder<Neighborhood> {

    public static final int NUM_INPUTS = FeatureExtractor.NUM_FEATURES;
    public static final int NUM_OUTPUTS = 1;   // one output: the typical home value

    // The expected CSV header: the eight neighborhood fields and the value to predict (medianHomeValueKusd).
    private static final List<String> HEADER = List.of(
            "crimeRate", "avgRooms", "oldBuildingsPercent", "distanceToEmploymentKm",
            "highwayAccessIndex", "pupilTeacherRatio", "nitrogenOxidesPpm", "lowerStatusPercent",
            "medianHomeValueKusd");

    // The value to predict: typical (median) home value, in thousands of US dollars (kUSD).
    private static final int TARGET_COLUMN = HEADER.size() - 1;

    // The network learns best when numbers are small, so the value to predict is divided by
    // 50 (a bit above the largest value in the data) before training,
    // and multiplied by it again after prediction.
    // A fixed number (instead of one calculated from the data) is simple, and works the same
    // in training and in prediction, so nothing extra has to be saved with the network.
    private static final double TARGET_MAX = 50;

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

    // Converts a neighborhood into the numbers the network uses (tensors).
    @Override
    public Tensor1D toTensor(Neighborhood neighborhood) {
        NeighborhoodValidator.validate(neighborhood);
        return Tensor1D.of(FeatureExtractor.extract(neighborhood));
    }

    // Converts a CSV row into a neighborhood.
    // CSV columns: crimeRate, avgRooms, oldBuildingsPercent, distanceToEmploymentKm, highwayAccessIndex,
    // pupilTeacherRatio, nitrogenOxidesPpm, lowerStatusPercent.
    @Override
    public Neighborhood toRecord(String[] columns) {
        return new Neighborhood(
                Double.parseDouble(columns[0]),
                Double.parseDouble(columns[1]),
                Double.parseDouble(columns[2]),
                Double.parseDouble(columns[3]),
                Integer.parseInt(columns[4]),
                Double.parseDouble(columns[5]),
                Double.parseDouble(columns[6]),
                Double.parseDouble(columns[7]));
    }

    // The value to predict, put on the same small scale as the inputs (see NeighborhoodEncoder).
    private static float scaledTarget(String value) {
        double target = Double.parseDouble(value);
        if (!Double.isFinite(target) || target < 0) {
            throw new IllegalArgumentException("medianHomeValueKusd must be a number >= 0, was " + value);
        }
        return scaleTarget(target);
    }
}
