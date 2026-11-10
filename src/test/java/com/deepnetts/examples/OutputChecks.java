package com.deepnetts.examples;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

// Reads what an example printed and checks that the network is better than the baseline (the answer without a network).
final class OutputChecks {

    private static final Pattern ACCURACY = Pattern.compile("(?m)^\\s*Accuracy\\s+([\\d.]+)\\s*%");
    private static final Pattern BASELINE_ACCURACY = Pattern.compile("(?m)^\\s*Baseline\\s+([\\d.]+)\\s*%");
    private static final Pattern ERROR = Pattern.compile("(?m)^\\s*(?:MAE|Average error)\\s+([\\d.]+)");
    private static final Pattern BASELINE_ERROR = Pattern.compile("(?m)^\\s*Baseline\\s+([\\d.]+)\\s+[A-Za-z]");
    private static final Pattern PICKED = Pattern.compile("(?m)^\\s*Picked the fastest\\s+(\\d+) of");
    private static final Pattern BASELINE_PICKED = Pattern.compile("(?m)^\\s*Baseline\\s+(\\d+) of");

    private OutputChecks() {
    }

    // Empty when the network beats the baseline, or when the output has no baseline to compare with
    // (Hello World, Light switches). Otherwise it says what is wrong.
    static String problemWithBaseline(String output) {
        Double accuracy = number(ACCURACY, output);
        Double baselineAccuracy = number(BASELINE_ACCURACY, output);
        if (accuracy != null && baselineAccuracy != null) {
            return accuracy > baselineAccuracy ? "" : "accuracy " + accuracy + "% is not above the baseline " + baselineAccuracy + "%";
        }
        Double error = number(ERROR, output);
        Double baselineError = number(BASELINE_ERROR, output);
        if (error != null && baselineError != null) {
            return error < baselineError ? "" : "error " + error + " is not below the baseline error " + baselineError;
        }
        Double picked = number(PICKED, output);
        Double baselinePicked = number(BASELINE_PICKED, output);
        if (picked != null && baselinePicked != null) {
            return picked > baselinePicked ? "" : "the network picked the fastest " + picked.intValue()
                    + " times, not more than the baseline " + baselinePicked.intValue();
        }
        return "";
    }

    private static Double number(Pattern pattern, String output) {
        Matcher m = pattern.matcher(output);
        return m.find() ? Double.valueOf(m.group(1)) : null;
    }
}
