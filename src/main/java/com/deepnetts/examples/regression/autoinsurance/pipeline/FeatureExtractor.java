package com.deepnetts.examples.regression.autoinsurance.pipeline;

import com.deepnetts.examples.regression.autoinsurance.dto.PolicyGroup;
import com.deepnetts.examples.regression.autoinsurance.enums.Zone;

import deepnetts.data.DataSets;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.stream.IntStream;

// Turns a policy group into the numbers the network uses.
public final class FeatureExtractor {

    private static final String[] ZONE_NAMES =
            Arrays.stream(Zone.values()).map(Enum::name).toArray(String[]::new);

    // A vehicle make is a code from 1 to 9. The codes have no order (make 9 is not "more" than make 2),
    // so each one gets its own 0/1 input, like the zones.
    private static final String[] MAKE_NAMES =
            IntStream.rangeClosed(1, 9).mapToObj(String::valueOf).toArray(String[]::new);

    // Name and order of every input. The order must match extract().
    public static final List<String> FEATURE_NAMES = buildFeatureNames();

    public static final int NUM_FEATURES = FEATURE_NAMES.size();

    // The largest values in the data. Dividing by them puts every input roughly between 0 and 1,
    // so no input dominates.
    private static final double MAX_KILOMETRES_CLASS = 5;
    private static final double MAX_BONUS_CLASS      = 7;
    private static final double MAX_INSURED_YEARS    = 3500;

    private FeatureExtractor() {
    }

    public static float[] extract(PolicyGroup group) {
        float[] base = {
                (float) (group.kilometresClass() / MAX_KILOMETRES_CLASS),
                (float) (group.bonusClass() / MAX_BONUS_CLASS),
                // log1p(x) is log(1 + x). Most groups are small, a few are huge:
                // using log so the huge ones don't dominate.
                (float) (Math.log1p(group.insuredYears()) / Math.log1p(MAX_INSURED_YEARS))
        };
        // One 0/1 input per zone and one per vehicle make.
        float[] zone = DataSets.oneHotEncode(group.zone().name(), ZONE_NAMES);
        float[] make = DataSets.oneHotEncode(String.valueOf(group.vehicleMake()), MAKE_NAMES);
        return concat(concat(base, zone), make);
    }

    private static List<String> buildFeatureNames() {
        List<String> names = new ArrayList<>(List.of(
                "kilometres_class", "bonus_class", "log_insured_years"));
        for (String zone : ZONE_NAMES) {
            names.add("zone_" + zone.toLowerCase(Locale.ROOT));
        }
        for (String make : MAKE_NAMES) {
            names.add("make_" + make);
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
