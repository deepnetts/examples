package com.deepnetts.examples.regression.cpuperformance.pipeline;

import com.deepnetts.examples.regression.cpuperformance.dto.Cpu;

// Checks that a CPU has valid values.
public final class CpuValidator {

    // Allows for rounding: 1.0000001 still counts as 1.
    private static final float TOLERANCE = 1e-6f;

    private CpuValidator() {
    }

    // Same check, but the error says where the CPU came from.
    public static void validate(Cpu cpu, String source) {
        try {
            validate(cpu);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(source + ": " + e.getMessage(), e);
        }
    }

    // Throws an error naming the broken rule and the bad value.
    public static void validate(Cpu cpu) {
        if (cpu == null) {
            throw new IllegalArgumentException("cpu must not be null");
        }
        if (cpu.machineCycleNs() <= 0) {
            reject("machineCycleNs must be > 0", cpu.machineCycleNs());
        }
        if (cpu.minMemoryKb() < 0) {
            reject("minMemoryKb must be >= 0", cpu.minMemoryKb());
        }
        if (cpu.maxMemoryKb() < cpu.minMemoryKb()) {
            reject("maxMemoryKb must not be smaller than minMemoryKb (" + cpu.minMemoryKb() + ")", cpu.maxMemoryKb());
        }
        if (cpu.cacheMemoryKb() < 0) {
            reject("cacheMemoryKb must be >= 0", cpu.cacheMemoryKb());
        }
        if (cpu.minChannels() < 0) {
            reject("minChannels must be >= 0", cpu.minChannels());
        }
        if (cpu.maxChannels() < cpu.minChannels()) {
            reject("maxChannels must not be smaller than minChannels (" + cpu.minChannels() + ")", cpu.maxChannels());
        }
    }

    // Throws an error naming the broken rule and the bad value: "<rule>, was <value>".
    private static void reject(String rule, Object actualValue) {
        throw new IllegalArgumentException(rule + ", was " + actualValue);
    }

    // The network learned from inputs from 0 to 1 (each value divided by the largest one in the data).
    // Outside that range it can only guess, and nothing else in the output would show it.
    // Prints one warning for every such input of cpu. The example still runs.
    public static void warnIfOutsideData(Cpu cpu, String source) {
        float[] inputs = FeatureExtractor.extract(cpu);
        for (int i = 0; i < inputs.length; i++) {
            String name = FeatureExtractor.FEATURE_NAMES.get(i);
            if (inputs[i] > 1 + TOLERANCE) {
                warn(source, name, "above the largest value in the data");
            } else if (inputs[i] < -TOLERANCE) {
                warn(source, name, "below the smallest value in the data");
            }
        }
    }

    private static void warn(String source, String input, String where) {
        System.out.println("Warning: " + source + ": " + input + " is " + where
                + ", so the network can only guess. The example still runs.");
    }
}
