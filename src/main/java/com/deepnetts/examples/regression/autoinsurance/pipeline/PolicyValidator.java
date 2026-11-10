package com.deepnetts.examples.regression.autoinsurance.pipeline;

import com.deepnetts.examples.regression.autoinsurance.dto.PolicyGroup;

// Checks that a policy group has valid values.
public final class PolicyValidator {

    // Allows for rounding: 1.0000001 still counts as 1.
    private static final float TOLERANCE = 1e-6f;

    private static final int MIN_KILOMETRES_CLASS = 1;
    private static final int MAX_KILOMETRES_CLASS = 5;
    private static final int MIN_BONUS_CLASS = 1;
    private static final int MAX_BONUS_CLASS = 7;
    private static final int MIN_VEHICLE_MAKE = 1;
    private static final int MAX_VEHICLE_MAKE = 9;

    private PolicyValidator() {
    }

    // Same check, but the error says where the policy group came from.
    public static void validate(PolicyGroup group, String source) {
        try {
            validate(group);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(source + ": " + e.getMessage(), e);
        }
    }

    // Throws an error naming the broken rule and the bad value.
    public static void validate(PolicyGroup group) {
        if (group == null) {
            throw new IllegalArgumentException("group must not be null");
        }
        if (group.zone() == null) {
            throw new IllegalArgumentException("zone must not be null");
        }
        if (group.kilometresClass() < MIN_KILOMETRES_CLASS || group.kilometresClass() > MAX_KILOMETRES_CLASS) {
            reject("kilometresClass must be between " + MIN_KILOMETRES_CLASS + " and " + MAX_KILOMETRES_CLASS, group.kilometresClass());
        }
        if (group.bonusClass() < MIN_BONUS_CLASS || group.bonusClass() > MAX_BONUS_CLASS) {
            reject("bonusClass must be between " + MIN_BONUS_CLASS + " and " + MAX_BONUS_CLASS, group.bonusClass());
        }
        if (group.vehicleMake() < MIN_VEHICLE_MAKE || group.vehicleMake() > MAX_VEHICLE_MAKE) {
            reject("vehicleMake must be between " + MIN_VEHICLE_MAKE + " and " + MAX_VEHICLE_MAKE, group.vehicleMake());
        }
        if (!Double.isFinite(group.insuredYears()) || group.insuredYears() <= 0) {
            reject("insuredYears must be a number > 0", group.insuredYears());
        }
    }

    // Throws an error naming the broken rule and the bad value: "<rule>, was <value>".
    private static void reject(String rule, Object actualValue) {
        throw new IllegalArgumentException(rule + ", was " + actualValue);
    }

    // The network learned from inputs from 0 to 1 (each value divided by the largest one in the data).
    // Outside that range it can only guess, and nothing else in the output would show it.
    // Prints one warning for every such input of group. The example still runs.
    public static void warnIfOutsideData(PolicyGroup group, String source) {
        float[] inputs = FeatureExtractor.extract(group);
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
