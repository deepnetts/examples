package com.deepnetts.examples.classification.creditcardfraud.pipeline;

import com.deepnetts.examples.classification.creditcardfraud.dto.Transaction;
import com.deepnetts.examples.classification.creditcardfraud.enums.MerchantCategory;
import com.deepnetts.examples.util.CsvEncoder;

import deepnetts.tensor.Tensor1D;

import java.util.List;
import java.util.Locale;

// Turns data into the numbers the network learns from:
// a CSV row becomes a Transaction (toRecord), a Transaction becomes the network inputs (toTensor),
// and the label column becomes the correct answer (toTarget). load() (from CsvEncoder) reads the whole CSV this way.
public final class TransactionEncoder implements CsvEncoder<Transaction> {

    // The expected CSV header: the four transaction fields and the label (is_fraud).
    private static final List<String> HEADER = List.of(
            "amount", "category", "hourOfDay", "distanceFromHomeKm", "is_fraud");

    // The label is the correct answer the network learns: 1 = fraud, 0 = legitimate.
    private static final int LABEL_COLUMN = HEADER.size() - 1;

    public static final int NUM_INPUTS = FeatureExtractor.NUM_FEATURES;
    public static final int NUM_OUTPUTS = 1;   // one output: the fraud probability

    @Override
    public List<String> header() {
        return HEADER;
    }

    @Override
    public List<String> inputNames() {
        return FeatureExtractor.FEATURE_NAMES;
    }

    @Override
    public List<String> outputNames() {
        return List.of(HEADER.get(LABEL_COLUMN));
    }

    // Converts a transaction into the numbers the network uses (a tensor: Deep Netts' array of numbers).
    @Override
    public Tensor1D toTensor(Transaction transaction) {
        TransactionValidator.validate(transaction);
        return Tensor1D.of(FeatureExtractor.extract(transaction));
    }

    // Converts a CSV row into a transaction.
    // CSV columns: amount, category, hourOfDay, distanceFromHomeKm.
    @Override
    public Transaction toRecord(String[] columns) {
        return new Transaction(
                Double.parseDouble(columns[0]),
                merchantCategory(columns[1]),
                Integer.parseInt(columns[2]),
                Double.parseDouble(columns[3]));
    }

    // The correct answer of a CSV row: 1 = fraud, 0 = legitimate.
    @Override
    public Tensor1D toTarget(String[] columns) {
        String label = columns[LABEL_COLUMN];
        return switch (label) {
            case "1" -> Tensor1D.of(new float[] {1f});
            case "0" -> Tensor1D.of(new float[] {0f});
            default -> throw new IllegalArgumentException("is_fraud must be 0 or 1, was " + label);
        };
    }

    // The merchant category with this name, in any letter case ("shopping_net" = SHOPPING_NET).
    private static MerchantCategory merchantCategory(String name) {
        try {
            return MerchantCategory.valueOf(name.toUpperCase(Locale.ROOT));
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("unknown merchant category: " + name);
        }
    }
}
