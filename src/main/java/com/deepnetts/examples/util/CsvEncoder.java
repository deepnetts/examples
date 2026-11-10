package com.deepnetts.examples.util;

import deepnetts.data.DataSetItem;
import deepnetts.data.MLDataItem;
import deepnetts.data.TabularDataSet;
import deepnetts.tensor.Tensor1D;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Consumer;

// The encoder of an example that learns from a CSV file.
// It turns a CSV row into a record (ToRecord) and a record into the network inputs (ToTensor),
// and says what the correct answer of a row is (toTarget). With that, load() reads the whole CSV,
// so an example doesn't need a separate loader.
public interface CsvEncoder<R extends Record> extends ToRecord<R>, ToTensor<R> {

    // The expected CSV header: the input columns, then the column(s) with the correct answer.
    List<String> header();

    // The names of the network inputs, in the order of toTensor().
    List<String> inputNames();

    // The names of the network outputs, in the order of toTarget().
    List<String> outputNames();

    // The correct answer of one CSV row, as the numbers the network learns to give.
    Tensor1D toTarget(String[] columns);

    // Reads the CSV file into a dataset the network can train on: each row becomes a record,
    // then the network inputs, together with its correct answer.
    default TabularDataSet<MLDataItem> load(String csvPath) throws IOException {
        TabularDataSet<MLDataItem> data = new TabularDataSet<>(inputNames().size(), outputNames().size());
        List<String> names = new ArrayList<>(inputNames());
        names.addAll(outputNames());
        data.setColumnNames(names.toArray(String[]::new));

        forEachRow(csvPath, columns -> data.add(new DataSetItem(toTensor(toRecord(columns)), toTarget(columns))));
        return data;
    }

    // Reads the CSV file and gives every row, split into columns, to rowAction.
    // The first line must be exactly the expected header; empty lines are skipped.
    // An error names the file and the line, for example "creditcard-fraud.csv, line 7: hourOfDay must be ...".
    default void forEachRow(String csvPath, Consumer<String[]> rowAction) throws IOException {
        Path path = Path.of(csvPath);
        if (!Files.exists(path)) {
            throw new FileNotFoundException("File not found: " + path.toAbsolutePath()
                    + " (run the example from the project root folder)");
        }
        List<String> lines = Files.readAllLines(path);
        if (lines.isEmpty() || !columns(lines.getFirst()).equals(header())) {
            throw new IOException(path.getFileName() + ": expected the header " + String.join(",", header()));
        }

        for (int row = 1; row < lines.size(); row++) {   // row 0 is the header
            if (lines.get(row).isBlank()) {
                continue;
            }
            try {
                String[] columns = columns(lines.get(row)).toArray(String[]::new);
                if (columns.length != header().size()) {
                    throw new IllegalArgumentException("expected " + header().size() + " columns, found " + columns.length);
                }
                rowAction.accept(columns);
            } catch (RuntimeException e) {
                throw new IOException(path.getFileName() + ", line " + (row + 1) + ": " + e.getMessage(), e);
            }
        }
    }

    // The columns of one CSV line, trimmed.
    private static List<String> columns(String line) {
        return Arrays.stream(line.split(",", -1)).map(String::trim).toList();
    }
}
