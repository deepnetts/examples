package com.deepnetts.examples.classification.irisflower.pipeline;

import com.deepnetts.examples.classification.irisflower.dto.Flower;
import com.deepnetts.examples.classification.irisflower.enums.Species;
import com.deepnetts.examples.util.CsvEncoder;

import deepnetts.tensor.Tensor1D;

import java.util.List;
import java.util.Locale;
import java.util.stream.Stream;

// Bridge between our records and the network, in both directions, and between the CSV file and the network.
public final class FlowerEncoder implements CsvEncoder<Flower> {

    public static final int NUM_INPUTS = FeatureExtractor.NUM_FEATURES;
    public static final int NUM_OUTPUTS = Species.values().length;   // one output per species: its probability

    // The expected CSV header: the four flower measurements and the label (species).
    private static final List<String> HEADER = List.of(
            "sepalLengthCm", "sepalWidthCm", "petalLengthCm", "petalWidthCm", "species");

    // The label: setosa, versicolor or virginica.
    private static final int LABEL_COLUMN = HEADER.size() - 1;

    @Override
    public List<String> header() {
        return HEADER;
    }

    @Override
    public List<String> inputNames() {
        return FeatureExtractor.FEATURE_NAMES;
    }

    // One output per species, named after it: setosa, versicolor, virginica.
    @Override
    public List<String> outputNames() {
        return Stream.of(Species.values()).map(s -> s.name().toLowerCase(Locale.ROOT)).toList();
    }

    // The correct answer of a CSV row: the species, as 3 numbers (see toTarget(Species)).
    @Override
    public Tensor1D toTarget(String[] columns) {
        return toTarget(enumValue(Species.class, columns[LABEL_COLUMN], "species"));
    }

    // Converts a flower into the numbers the network uses (tensors).
    @Override
    public Tensor1D toTensor(Flower flower) {
        FlowerValidator.validate(flower);
        return Tensor1D.of(FeatureExtractor.extract(flower));
    }

    // Converts a CSV row into a flower.
    // CSV columns: sepalLengthCm, sepalWidthCm, petalLengthCm, petalWidthCm (the species is read by toTarget).
    @Override
    public Flower toRecord(String[] columns) {
        return new Flower(
                Double.parseDouble(columns[0]),
                Double.parseDouble(columns[1]),
                Double.parseDouble(columns[2]),
                Double.parseDouble(columns[3]));
    }

    // The correct answer the network learns: 3 numbers, a 1 at the place of the species and 0 elsewhere.
    // This is called one-hot encoding.
    public Tensor1D toTarget(Species species) {
        float[] target = new float[NUM_OUTPUTS];
        target[species.ordinal()] = 1f;
        return Tensor1D.of(target);
    }

    // The network gives 3 probabilities, one per species. The answer is the species with the highest one.
    public Species toSpecies(float[] outputs) {
        int best = 0;
        for (int i = 1; i < outputs.length; i++) {
            if (outputs[i] > outputs[best]) {
                best = i;
            }
        }
        return Species.values()[best];
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
