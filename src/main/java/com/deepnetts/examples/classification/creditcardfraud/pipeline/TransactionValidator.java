package com.deepnetts.examples.classification.creditcardfraud.pipeline;

import com.deepnetts.examples.classification.creditcardfraud.dto.Transaction;

// Checks that a transaction has valid values.
public final class TransactionValidator {

    private static final int FIRST_HOUR_OF_DAY = 0;
    private static final int LAST_HOUR_OF_DAY = 23;

    private TransactionValidator() {
    }

    // Same check, but the error says where the transaction came from.
    public static void validate(Transaction transaction, String source) {
        try {
            validate(transaction);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(source + ": " + e.getMessage(), e);
        }
    }

    // Throws an error naming the broken rule and the bad value.
    public static void validate(Transaction transaction) {
        if (transaction == null) {
            throw new IllegalArgumentException("transaction must not be null");
        }
        if (transaction.merchantCategory() == null) {
            throw new IllegalArgumentException("merchantCategory must not be null");
        }
        if (!isNonNegativeNumber(transaction.amount())) {
            reject("amount must be a number >= 0", transaction.amount());
        }
        if (transaction.hourOfDay() < FIRST_HOUR_OF_DAY || transaction.hourOfDay() > LAST_HOUR_OF_DAY) {
            reject("hourOfDay must be between " + FIRST_HOUR_OF_DAY + " and " + LAST_HOUR_OF_DAY, transaction.hourOfDay());
        }
        if (!isNonNegativeNumber(transaction.distanceFromHomeKm())) {
            reject("distanceFromHomeKm must be a number >= 0", transaction.distanceFromHomeKm());
        }
    }

    // Rejects "not a number" (NaN) and infinity.
    private static boolean isNonNegativeNumber(double value) {
        return Double.isFinite(value) && value >= 0;
    }

    // Throws an error naming the broken rule and the bad value: "<rule>, was <value>".
    private static void reject(String rule, Object actualValue) {
        throw new IllegalArgumentException(rule + ", was " + actualValue);
    }
}
