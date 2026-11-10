package com.deepnetts.examples.regression.taxifare.pipeline;

import com.deepnetts.examples.regression.taxifare.dto.Ride;

import java.util.List;

// Turns a ride into the numbers the network uses. Each of these numbers is called a feature.
public final class FeatureExtractor {

    // The largest values in the data (rounded up). Dividing by them puts every input between 0 and 1,
    // so a big number (minutes) doesn't count more than a small one (km) just because it is bigger.
    static final double MAX_DISTANCE_KM = 30;
    static final double MAX_DURATION_MINUTES = 90;

    // Name and order of every input. The order must match extract().
    public static final List<String> FEATURE_NAMES = List.of("distance_km", "duration_minutes");

    public static final int NUM_FEATURES = FEATURE_NAMES.size();

    private FeatureExtractor() {
    }

    public static float[] extract(Ride ride) {
        return new float[] {
                (float) (ride.distanceKm() / MAX_DISTANCE_KM),
                (float) (ride.durationMinutes() / MAX_DURATION_MINUTES)
        };
    }
}
