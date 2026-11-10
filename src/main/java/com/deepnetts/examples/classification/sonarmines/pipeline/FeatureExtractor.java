package com.deepnetts.examples.classification.sonarmines.pipeline;

import com.deepnetts.examples.classification.sonarmines.dto.SonarContact;
import com.deepnetts.examples.classification.sonarmines.enums.SeabedType;

import deepnetts.data.DataSets;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

// Turns a sonar contact into the numbers the network uses.
public final class FeatureExtractor {

    private static final String[] SEABED_NAMES =
            Arrays.stream(SeabedType.values()).map(Enum::name).toArray(String[]::new);

    // Name and order of every input. The order must match extract().
    public static final List<String> FEATURE_NAMES = buildFeatureNames();

    public static final int NUM_FEATURES = FEATURE_NAMES.size();

    private FeatureExtractor() {
    }

    public static float[] extract(SonarContact contact) {
        float[] base = {
                (float) contact.rangeM(),
                contact.aspectAngleDeg(),
                (float) contact.echoLevelDb(),
                (float) contact.echoDurationMs(),
                (float) contact.spectralCentroidKhz(),
                // log1p(x) is log(1 + x). Most shadows are short, a few are very long:
                // using log so the very long ones don't dominate.
                (float) Math.log1p(contact.shadowLengthM())
        };
        // One 0/1 input per seabed type.
        return concat(base, DataSets.oneHotEncode(contact.seabedType().name(), SEABED_NAMES));
    }

    private static List<String> buildFeatureNames() {
        List<String> names = new ArrayList<>(List.of(
                "range_m", "aspect_angle_deg", "echo_level_db", "echo_duration_ms",
                "spectral_centroid_khz", "log_shadow_length_m"));
        for (String seabed : SEABED_NAMES) {
            names.add("seabed_" + seabed.toLowerCase(Locale.ROOT));
        }
        return List.copyOf(names);
    }

    private static float[] concat(float[] a, float[] b) {
        float[] result = new float[a.length + b.length];
        System.arraycopy(a, 0, result, 0, a.length);
        System.arraycopy(b, 0, result, a.length, b.length);
        return result;
    }
}
