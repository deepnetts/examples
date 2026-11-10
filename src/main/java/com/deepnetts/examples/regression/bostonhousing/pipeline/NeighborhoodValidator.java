package com.deepnetts.examples.regression.bostonhousing.pipeline;

import com.deepnetts.examples.regression.bostonhousing.dto.Neighborhood;

// Checks that a neighborhood has valid values.
public final class NeighborhoodValidator {

    // Allows for rounding: 1.0000001 still counts as 1.
    private static final float TOLERANCE = 1e-6f;

    private static final double MAX_PERCENT = 100;

    private NeighborhoodValidator() {
    }

    // Same check, but the error says where the neighborhood came from.
    public static void validate(Neighborhood neighborhood, String source) {
        try {
            validate(neighborhood);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(source + ": " + e.getMessage(), e);
        }
    }

    // Throws an error naming the broken rule and the bad value.
    public static void validate(Neighborhood neighborhood) {
        if (neighborhood == null) {
            throw new IllegalArgumentException("neighborhood must not be null");
        }
        if (!isNonNegativeNumber(neighborhood.crimeRate())) {
            reject("crimeRate must be a number >= 0", neighborhood.crimeRate());
        }
        if (!isNonNegativeNumber(neighborhood.avgRooms()) || neighborhood.avgRooms() == 0) {
            reject("avgRooms must be a number > 0", neighborhood.avgRooms());
        }
        validatePercent("oldBuildingsPercent", neighborhood.oldBuildingsPercent());
        if (!isNonNegativeNumber(neighborhood.distanceToEmploymentKm())) {
            reject("distanceToEmploymentKm must be a number >= 0", neighborhood.distanceToEmploymentKm());
        }
        if (neighborhood.highwayAccessIndex() < 1) {
            reject("highwayAccessIndex must be >= 1", neighborhood.highwayAccessIndex());
        }
        if (!isNonNegativeNumber(neighborhood.pupilTeacherRatio()) || neighborhood.pupilTeacherRatio() == 0) {
            reject("pupilTeacherRatio must be a number > 0", neighborhood.pupilTeacherRatio());
        }
        if (!isNonNegativeNumber(neighborhood.nitrogenOxidesPpm()) || neighborhood.nitrogenOxidesPpm() > 1) {
            reject("nitrogenOxidesPpm must be a number between 0 and 1", neighborhood.nitrogenOxidesPpm());
        }
        validatePercent("lowerStatusPercent", neighborhood.lowerStatusPercent());
    }

    private static void validatePercent(String name, double value) {
        if (!isNonNegativeNumber(value) || value > MAX_PERCENT) {
            reject(name + " must be a number between 0 and " + (int) MAX_PERCENT, value);
        }
    }

    // Rejects "not a number" (NaN), infinity and negative values.
    private static boolean isNonNegativeNumber(double value) {
        return Double.isFinite(value) && value >= 0;
    }

    // Throws an error naming the broken rule and the bad value: "<rule>, was <value>".
    private static void reject(String rule, Object actualValue) {
        throw new IllegalArgumentException(rule + ", was " + actualValue);
    }

    // The network learned from inputs from 0 to 1 (each value divided by the largest one in the data).
    // Outside that range it can only guess, and nothing else in the output would show it.
    // Prints one warning for every such input of neighborhood. The example still runs.
    public static void warnIfOutsideData(Neighborhood neighborhood, String source) {
        float[] inputs = FeatureExtractor.extract(neighborhood);
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
