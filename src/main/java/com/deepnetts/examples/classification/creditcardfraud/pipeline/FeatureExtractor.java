package com.deepnetts.examples.classification.creditcardfraud.pipeline;

import com.deepnetts.examples.classification.creditcardfraud.dto.Transaction;
import com.deepnetts.examples.classification.creditcardfraud.enums.MerchantCategory;

import deepnetts.data.DataSets;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

// Turns a transaction into the numbers the network uses. Each of these numbers is called a feature.
public final class FeatureExtractor {

    private static final String[] CATEGORY_NAMES =
            Arrays.stream(MerchantCategory.values()).map(Enum::name).toArray(String[]::new);

    // Name and order of every input. The order must match extract().
    public static final List<String> FEATURE_NAMES = buildFeatureNames();

    public static final int NUM_FEATURES = FEATURE_NAMES.size();

    private FeatureExtractor() {
    }

    public static float[] extract(Transaction transaction) {
        double hourAngle = 2 * Math.PI * transaction.hourOfDay() / 24.0;
        float[] base = {
                (float) Math.log1p(transaction.amount()),                // using log so very large amounts don't dominate
                (float) ((Math.sin(hourAngle) + 1) / 2),                 // using sin/cos so 23:00 and 00:00 are close together
                (float) ((Math.cos(hourAngle) + 1) / 2),
                transaction.hourOfDay() < 6 ? 1f : 0f,                   // night (0-5 AM): in this data, the hours with the most fraud
                (float) Math.log1p(transaction.distanceFromHomeKm())     // using log so very large distances don't dominate
        };
        // One 0/1 input per merchant category.
        return concat(base, DataSets.oneHotEncode(transaction.merchantCategory().name(), CATEGORY_NAMES));
    }

    private static List<String> buildFeatureNames() {
        List<String> names = new ArrayList<>(List.of(
                "log_amount", "hour_sin", "hour_cos", "is_night", "log_distance_from_home"));
        for (String category : CATEGORY_NAMES) {
            names.add("category_" + category.toLowerCase(Locale.ROOT));
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
