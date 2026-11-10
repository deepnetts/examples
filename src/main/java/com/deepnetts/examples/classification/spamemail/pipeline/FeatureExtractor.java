package com.deepnetts.examples.classification.spamemail.pipeline;

import com.deepnetts.examples.classification.spamemail.dto.Email;

import java.util.List;

// Turns an email into the numbers the network uses.
public final class FeatureExtractor {

    // Name and order of every input. The order must match extract().
    public static final List<String> FEATURE_NAMES = List.of(
            "log_free_word_percent", "log_money_word_percent", "log_remove_word_percent",
            "log_your_word_percent", "log_exclamation_char_percent", "log_dollar_char_percent",
            "log_longest_capital_run", "log_total_capital_letters");

    public static final int NUM_FEATURES = FEATURE_NAMES.size();

    private FeatureExtractor() {
    }

    public static float[] extract(Email email) {
        return new float[] {
                log((float) email.freeWordPercent()),
                log((float) email.moneyWordPercent()),
                log((float) email.removeWordPercent()),
                log((float) email.yourWordPercent()),
                log((float) email.exclamationCharPercent()),
                log((float) email.dollarCharPercent()),
                log(email.longestCapitalRun()),
                log(email.totalCapitalLetters())
        };
    }

    // log1p(x) is log(1 + x). Most emails have values near 0, but a few have huge ones
    // (for example 10 000 capital letters in a row): using log so those don't dominate.
    // It also works for exactly 0, which is very common here.
    public static float log(float value) {
        return (float) Math.log1p(value);
    }
}
