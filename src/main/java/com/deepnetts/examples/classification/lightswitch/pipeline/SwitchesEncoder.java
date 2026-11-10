package com.deepnetts.examples.classification.lightswitch.pipeline;

import com.deepnetts.examples.classification.lightswitch.dto.Switches;
import com.deepnetts.examples.util.CsvEncoder;

import deepnetts.tensor.Tensor1D;

import java.util.List;

// Turns data into the numbers the network learns from:
// a CSV row becomes Switches (toRecord), Switches become the network inputs (toTensor),
// and the label column becomes the correct answer (toTarget). load() (from CsvEncoder) reads the whole CSV this way.
public final class SwitchesEncoder implements CsvEncoder<Switches> {

    // The expected CSV header: the two switches and the label (lightOn).
    private static final List<String> HEADER = List.of("bottomSwitchUp", "topSwitchUp", "lightOn");

    // The label is the correct answer the network learns: 1 = the light is on, 0 = off.
    private static final int LABEL_COLUMN = HEADER.size() - 1;

    public static final int NUM_INPUTS = 2;    // one 0/1 number per switch
    public static final int NUM_OUTPUTS = 1;   // one output: the probability that the light is on

    @Override
    public List<String> header() {
        return HEADER;
    }

    @Override
    public List<String> inputNames() {
        return List.of(HEADER.get(0), HEADER.get(1));
    }

    @Override
    public List<String> outputNames() {
        return List.of(HEADER.get(LABEL_COLUMN));
    }

    // Converts the switches into the numbers the network uses (a tensor: Deep Netts' array of numbers):
    // 1 for a switch that is up, 0 for one that is down.
    @Override
    public Tensor1D toTensor(Switches switches) {
        if (switches == null) {
            throw new IllegalArgumentException("switches must not be null");
        }
        return Tensor1D.of(new float[] {switches.bottomUp() ? 1f : 0f, switches.topUp() ? 1f : 0f});
    }

    // Converts a CSV row into switches.
    // CSV columns: bottomSwitchUp, topSwitchUp (1 = up, 0 = down).
    @Override
    public Switches toRecord(String[] columns) {
        return new Switches(flag(columns[0], HEADER.get(0)), flag(columns[1], HEADER.get(1)));
    }

    // The correct answer of a CSV row: 1 = on, 0 = off.
    @Override
    public Tensor1D toTarget(String[] columns) {
        return Tensor1D.of(new float[] {flag(columns[LABEL_COLUMN], HEADER.get(LABEL_COLUMN)) ? 1f : 0f});
    }

    // "1" = true, "0" = false.
    private static boolean flag(String value, String column) {
        return switch (value) {
            case "1" -> true;
            case "0" -> false;
            default -> throw new IllegalArgumentException(column + " must be 0 or 1, was " + value);
        };
    }
}
