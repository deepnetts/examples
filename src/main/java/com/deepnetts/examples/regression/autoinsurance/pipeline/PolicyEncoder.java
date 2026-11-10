package com.deepnetts.examples.regression.autoinsurance.pipeline;

import com.deepnetts.examples.regression.autoinsurance.dto.PolicyGroup;
import com.deepnetts.examples.regression.autoinsurance.enums.Zone;
import com.deepnetts.examples.util.CsvEncoder;

import deepnetts.tensor.Tensor1D;

import java.util.List;

// Bridge between our records and the network, and between the CSV file and the network.
public final class PolicyEncoder implements CsvEncoder<PolicyGroup> {

    public static final int NUM_INPUTS = FeatureExtractor.NUM_FEATURES;
    public static final int NUM_OUTPUTS = 1;   // one output: the payment per insured year

    // The expected CSV header: the five policy group fields and the value to predict (paymentPerInsuredYearKsek).
    private static final List<String> HEADER = List.of(
            "kilometresClass", "zone", "bonusClass", "vehicleMake",
            "insuredYears", "paymentPerInsuredYearKsek");

    // The value to predict: payment per insured year, in thousands of Swedish kronor (kSEK).
    private static final int TARGET_COLUMN = HEADER.size() - 1;

    // The network learns best when numbers are small, so the value to predict is divided by
    // 32 (a bit above the largest value in the data) before training,
    // and multiplied by it again after prediction.
    // A fixed number (instead of one calculated from the data) is simple, and works the same
    // in training and in prediction, so nothing extra has to be saved with the network.
    private static final double TARGET_MAX = 32;

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

    // Converts a policy group into the numbers the network uses (tensors).
    @Override
    public Tensor1D toTensor(PolicyGroup group) {
        PolicyValidator.validate(group);
        return Tensor1D.of(FeatureExtractor.extract(group));
    }

    // Converts a CSV row into a policy group.
    // CSV columns: kilometresClass, zone, bonusClass, vehicleMake, insuredYears.
    @Override
    public PolicyGroup toRecord(String[] columns) {
        return new PolicyGroup(
                Integer.parseInt(columns[0]),
                zone(columns[1]),
                Integer.parseInt(columns[2]),
                Integer.parseInt(columns[3]),
                Double.parseDouble(columns[4]));
    }

    // The CSV writes zones like "CitySmall"; the enum writes CITY_SMALL. Ignoring case and underscores makes them equal.
    private static Zone zone(String name) {
        for (Zone zone : Zone.values()) {
            if (zone.name().replace("_", "").equalsIgnoreCase(name)) {
                return zone;
            }
        }
        throw new IllegalArgumentException("unknown zone: " + name);
    }

    // The value to predict, put on the same small scale as the inputs (see PolicyEncoder).
    private static float scaledTarget(String value) {
        double target = Double.parseDouble(value);
        if (!Double.isFinite(target) || target < 0) {
            throw new IllegalArgumentException("paymentPerInsuredYearKsek must be a number >= 0, was " + value);
        }
        return scaleTarget(target);
    }
}
