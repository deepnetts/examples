package com.deepnetts.examples.classification.spamemail.pipeline;

import deepnetts.data.DataSetItem;
import deepnetts.data.MLDataItem;
import deepnetts.data.TabularDataSet;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

// The check of the CSV file that Deep Netts reads: the columns, the numbers and the label.
class EmailValidatorTest {

    private static final String[] COLUMNS = {
            "freeWordPercent", "moneyWordPercent", "removeWordPercent", "yourWordPercent",
            "exclamationCharPercent", "dollarCharPercent", "longestCapitalRun", "totalCapitalLetters", "is_spam"};

    private static TabularDataSet<MLDataItem> data(String[] columns, float[]... rows) {
        TabularDataSet<MLDataItem> data = new TabularDataSet<>(8, 1);
        data.setColumnNames(columns);
        for (float[] row : rows) {
            data.add(new DataSetItem(java.util.Arrays.copyOf(row, 8), new float[] {row[8]}));
        }
        return data;
    }

    @Test
    void acceptsValidRows() {
        float[] row = {0.5f, 0.2f, 0.3f, 1.4f, 0.9f, 0.2f, 120, 600, 1};

        assertDoesNotThrow(() -> EmailValidator.validateData(data(COLUMNS, row), "test.csv"));
    }

    @Test
    void namesTheFileAndTheRowOfANegativeValue() {
        float[] good = {0.5f, 0.2f, 0.3f, 1.4f, 0.9f, 0.2f, 120, 600, 1};
        float[] bad = {-5f, 0.2f, 0.3f, 1.4f, 0.9f, 0.2f, 120, 600, 1};

        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> EmailValidator.validateData(data(COLUMNS, good, bad), "test.csv"));

        assertEquals("test.csv, data row 2: freeWordPercent must be a number between 0 and 100, was -5.0", e.getMessage());
    }

    @Test
    void rejectsANumberThatIsNotValid() {
        float[] bad = {Float.NaN, 0.2f, 0.3f, 1.4f, 0.9f, 0.2f, 120, 600, 1};

        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> EmailValidator.validateData(data(COLUMNS, bad), "test.csv"));

        assertEquals("test.csv, data row 1: freeWordPercent must be a number, was NaN", e.getMessage());
    }

    @Test
    void rejectsALabelThatIsNotZeroOrOne() {
        float[] bad = {0.5f, 0.2f, 0.3f, 1.4f, 0.9f, 0.2f, 120, 600, 2};

        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> EmailValidator.validateData(data(COLUMNS, bad), "test.csv"));

        assertEquals("test.csv, data row 1: is_spam must be 0 or 1, was 2.0", e.getMessage());
    }

    @Test
    void rejectsColumnsInAnotherOrder() {
        String[] swapped = COLUMNS.clone();
        swapped[0] = "longestCapitalRun";
        swapped[6] = "freeWordPercent";
        float[] row = {0.5f, 0.2f, 0.3f, 1.4f, 0.9f, 0.2f, 120, 600, 1};

        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> EmailValidator.validateData(data(swapped, row), "test.csv"));

        assertEquals("test.csv, column 1 must be named freeWordPercent, was longestCapitalRun", e.getMessage());
    }
}
