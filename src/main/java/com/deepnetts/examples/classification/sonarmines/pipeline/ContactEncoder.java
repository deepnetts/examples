package com.deepnetts.examples.classification.sonarmines.pipeline;

import com.deepnetts.examples.classification.sonarmines.dto.SonarContact;
import com.deepnetts.examples.classification.sonarmines.enums.SeabedType;
import com.deepnetts.examples.util.CsvEncoder;

import deepnetts.tensor.Tensor1D;

import java.util.List;
import java.util.Locale;

// Bridge between our records and the network, and between the CSV file and the network.
public final class ContactEncoder implements CsvEncoder<SonarContact> {

    public static final int NUM_INPUTS = FeatureExtractor.NUM_FEATURES;
    public static final int NUM_OUTPUTS = 1;   // one output: the mine probability

    // The expected CSV header: the seven sonar contact fields and the label (is_mine).
    private static final List<String> HEADER = List.of(
            "rangeM", "aspectAngleDeg", "echoLevelDb", "echoDurationMs",
            "spectralCentroidKhz", "shadowLengthM", "seabedType", "is_mine");

    // The label: 1 = mine, 0 = not a mine.
    private static final int LABEL_COLUMN = HEADER.size() - 1;

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
        return List.of(HEADER.get(LABEL_COLUMN));
    }

    // The correct answer of a CSV row: 1 or 0.
    @Override
    public Tensor1D toTarget(String[] columns) {
        String label = columns[LABEL_COLUMN];
        return switch (label) {
            case "1" -> Tensor1D.of(new float[] {1f});
            case "0" -> Tensor1D.of(new float[] {0f});
            default -> throw new IllegalArgumentException("is_mine must be 0 or 1, was " + label);
        };
    }

    // Converts a sonar contact into the numbers the network uses (tensors).
    @Override
    public Tensor1D toTensor(SonarContact contact) {
        ContactValidator.validate(contact);
        return Tensor1D.of(FeatureExtractor.extract(contact));
    }

    // Converts a CSV row into a sonar contact.
    // CSV columns: rangeM, aspectAngleDeg, echoLevelDb, echoDurationMs, spectralCentroidKhz, shadowLengthM, seabedType.
    @Override
    public SonarContact toRecord(String[] columns) {
        return new SonarContact(
                Double.parseDouble(columns[0]),
                Integer.parseInt(columns[1]),
                Double.parseDouble(columns[2]),
                Double.parseDouble(columns[3]),
                Double.parseDouble(columns[4]),
                Double.parseDouble(columns[5]),
                enumValue(SeabedType.class, columns[6], "seabed type"));
    }

    // The enum constant with this name, in any letter case ("rock" = ROCK). what names it in the error.
    private static <E extends Enum<E>> E enumValue(Class<E> type, String value, String what) {
        try {
            return Enum.valueOf(type, value.toUpperCase(Locale.ROOT));
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("unknown " + what + ": " + value);
        }
    }
}
