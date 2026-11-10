package com.deepnetts.examples.classification.sonarmines.pipeline;

import com.deepnetts.examples.classification.sonarmines.dto.SonarContact;

// Checks that a sonar contact has valid values.
public final class ContactValidator {

    private static final int MIN_ANGLE = 0;
    private static final int MAX_ANGLE = 180;

    private ContactValidator() {
    }

    // Same check, but the error says where the contact came from.
    public static void validate(SonarContact contact, String source) {
        try {
            validate(contact);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(source + ": " + e.getMessage(), e);
        }
    }

    // Throws an error naming the broken rule and the bad value.
    public static void validate(SonarContact contact) {
        if (contact == null) {
            throw new IllegalArgumentException("contact must not be null");
        }
        if (contact.seabedType() == null) {
            throw new IllegalArgumentException("seabedType must not be null");
        }
        if (!isNonNegativeNumber(contact.rangeM())) {
            reject("rangeM must be a number >= 0", contact.rangeM());
        }
        if (contact.aspectAngleDeg() < MIN_ANGLE || contact.aspectAngleDeg() > MAX_ANGLE) {
            reject("aspectAngleDeg must be between " + MIN_ANGLE + " and " + MAX_ANGLE, contact.aspectAngleDeg());
        }
        if (!Double.isFinite(contact.echoLevelDb())) {   // negative values are normal for decibels
            reject("echoLevelDb must be a number", contact.echoLevelDb());
        }
        if (!isNonNegativeNumber(contact.echoDurationMs())) {
            reject("echoDurationMs must be a number >= 0", contact.echoDurationMs());
        }
        if (!isNonNegativeNumber(contact.spectralCentroidKhz())) {
            reject("spectralCentroidKhz must be a number >= 0", contact.spectralCentroidKhz());
        }
        if (!isNonNegativeNumber(contact.shadowLengthM())) {
            reject("shadowLengthM must be a number >= 0", contact.shadowLengthM());
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
