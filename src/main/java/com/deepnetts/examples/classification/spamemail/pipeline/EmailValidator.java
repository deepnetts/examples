package com.deepnetts.examples.classification.spamemail.pipeline;

import com.deepnetts.examples.classification.spamemail.dto.Email;

import deepnetts.data.MLDataItem;
import deepnetts.data.TabularDataSet;

import java.lang.reflect.RecordComponent;

// Checks that an email, and every row of the CSV file, has valid values.
public final class EmailValidator {

    private static final double MAX_PERCENT = 100;

    private EmailValidator() {
    }

    // Same check, but the error says where the email came from.
    public static void validate(Email email, String source) {
        try {
            validate(email);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(source + ": " + e.getMessage(), e);
        }
    }

    // Checks the CSV that Deep Netts has read, before the log scale is applied: the columns are the fields of Email
    // in the same order (FeatureExtractor depends on that order), every row has valid values and the label is 0 or 1.
    // The error names the file and the row, for example "spam-email.csv, data row 2: freeWordPercent must be ..., was -5.0".
    public static void validateData(TabularDataSet<MLDataItem> data, String source) {
        RecordComponent[] fields = Email.class.getRecordComponents();
        try {
            String[] columns = data.getColumnNames();
            if (columns.length != fields.length + 1) {
                reject("the header must have " + (fields.length + 1) + " columns", columns.length);
            }
            for (int i = 0; i < fields.length; i++) {
                if (!fields[i].getName().equals(columns[i])) {
                    reject("column " + (i + 1) + " must be named " + fields[i].getName(), columns[i]);
                }
            }
            int row = 0;
            for (MLDataItem item : data) {
                row++;
                try {
                    validateRow(item, fields);
                } catch (IllegalArgumentException e) {
                    throw new IllegalArgumentException("data row " + row + ": " + e.getMessage(), e);
                }
            }
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(source + ", " + e.getMessage(), e);
        }
    }

    private static void validateRow(MLDataItem item, RecordComponent[] fields) {
        float[] values = item.getInput().getValues();
        for (int i = 0; i < fields.length; i++) {
            if (!Float.isFinite(values[i])) {
                reject(fields[i].getName() + " must be a number", values[i]);
            }
        }
        // The same rules as for your email.
        validate(new Email(values[0], values[1], values[2], values[3], values[4], values[5], (int) values[6], (int) values[7]));
        float label = item.getTargetOutput().getValues()[0];
        if (label != 0f && label != 1f) {
            reject("is_spam must be 0 or 1", label);
        }
    }

    // Throws an error naming the broken rule and the bad value.
    public static void validate(Email email) {
        if (email == null) {
            throw new IllegalArgumentException("email must not be null");
        }
        validatePercent("freeWordPercent", email.freeWordPercent());
        validatePercent("moneyWordPercent", email.moneyWordPercent());
        validatePercent("removeWordPercent", email.removeWordPercent());
        validatePercent("yourWordPercent", email.yourWordPercent());
        validatePercent("exclamationCharPercent", email.exclamationCharPercent());
        validatePercent("dollarCharPercent", email.dollarCharPercent());
        if (email.longestCapitalRun() < 0) {
            reject("longestCapitalRun must be >= 0", email.longestCapitalRun());
        }
        if (email.totalCapitalLetters() < 0) {
            reject("totalCapitalLetters must be >= 0", email.totalCapitalLetters());
        }
        // The longest run of capitals is part of all the capital letters, so it can't be longer.
        if (email.longestCapitalRun() > email.totalCapitalLetters()) {
            reject("longestCapitalRun must not be larger than totalCapitalLetters", email.longestCapitalRun());
        }
    }

    private static void validatePercent(String name, double value) {
        if (!Double.isFinite(value) || value < 0 || value > MAX_PERCENT) {
            reject(name + " must be a number between 0 and " + (int) MAX_PERCENT, value);
        }
    }

    // Throws an error naming the broken rule and the bad value: "<rule>, was <value>".
    private static void reject(String rule, Object actualValue) {
        throw new IllegalArgumentException(rule + ", was " + actualValue);
    }
}
