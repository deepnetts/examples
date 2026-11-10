package com.deepnetts.examples.regression.taxifare.pipeline;

import com.deepnetts.examples.regression.taxifare.dto.Ride;

// Checks that a ride has valid values.
public final class RideValidator {

    private RideValidator() {
    }

    // Same check, but the error says where the ride came from.
    public static void validate(Ride ride, String source) {
        try {
            validate(ride);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(source + ": " + e.getMessage(), e);
        }
    }

    // Throws an error naming the broken rule and the bad value.
    public static void validate(Ride ride) {
        if (ride == null) {
            throw new IllegalArgumentException("ride must not be null");
        }
        if (!isPositiveNumber(ride.distanceKm())) {
            reject("distanceKm must be a number > 0", ride.distanceKm());
        }
        if (!isPositiveNumber(ride.durationMinutes())) {
            reject("durationMinutes must be a number > 0", ride.durationMinutes());
        }
    }

    // Rejects "not a number" (NaN), infinity, zero and negative values.
    private static boolean isPositiveNumber(double value) {
        return Double.isFinite(value) && value > 0;
    }

    // Throws an error naming the broken rule and the bad value: "<rule>, was <value>".
    private static void reject(String rule, Object actualValue) {
        throw new IllegalArgumentException(rule + ", was " + actualValue);
    }
}
