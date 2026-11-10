package com.deepnetts.examples.util;

import deepnetts.data.MLDataItem;
import deepnetts.data.TabularDataSet;
import deepnetts.tensor.Tensor1D;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

// Reading a CSV file with a CsvEncoder: the rows an example gets, and the errors a user sees for a broken file.
class CsvEncoderTest {

    // A tiny example: the input is a word (the network gets its length), the correct answer is a count.
    record Word(String text) {
    }

    static final class WordEncoder implements CsvEncoder<Word> {

        @Override
        public List<String> header() {
            return List.of("word", "count");
        }

        @Override
        public List<String> inputNames() {
            return List.of("length");
        }

        @Override
        public List<String> outputNames() {
            return List.of("count");
        }

        @Override
        public Word toRecord(String[] columns) {
            return new Word(columns[0]);
        }

        @Override
        public Tensor1D toTensor(Word word) {
            return Tensor1D.of(new float[] {word.text().length()});
        }

        @Override
        public Tensor1D toTarget(String[] columns) {
            return Tensor1D.of(new float[] {Integer.parseInt(columns[1])});
        }
    }

    private final WordEncoder encoder = new WordEncoder();

    @TempDir
    Path folder;

    private String write(String content) throws IOException {
        Path file = folder.resolve("test.csv");
        Files.writeString(file, content);
        return file.toString();
    }

    @Test
    void loadGivesTheInputsAndTheCorrectAnswerOfEveryRow() throws IOException {
        String csv = write("word,count\nabc,1\nde,2\n");

        TabularDataSet<MLDataItem> data = encoder.load(csv);

        assertEquals(2, data.size());
        assertArrayEquals(new float[] {3}, data.get(0).getInput().getValues());
        assertArrayEquals(new float[] {1}, data.get(0).getTargetOutput().getValues());
        assertArrayEquals(new float[] {2}, data.get(1).getInput().getValues());
        assertArrayEquals(new String[] {"length", "count"}, data.getColumnNames());
    }

    @Test
    void forEachRowGivesEveryRowTrimmedAndSkipsEmptyLines() throws IOException {
        String csv = write("word,count\n abc , 1\n\nde,2\n   \n");

        List<List<String>> rows = new ArrayList<>();
        encoder.forEachRow(csv, columns -> rows.add(List.of(columns)));

        assertEquals(List.of(List.of("abc", "1"), List.of("de", "2")), rows);
    }

    @Test
    void missingFileSaysWhereToRunFrom() {
        FileNotFoundException e = assertThrows(FileNotFoundException.class,
                () -> encoder.load(folder.resolve("missing.csv").toString()));

        assertTrue(e.getMessage().startsWith("File not found: "), e.getMessage());
        assertTrue(e.getMessage().endsWith("(run the example from the project root folder)"), e.getMessage());
    }

    @Test
    void wrongHeaderNamesTheExpectedOne() throws IOException {
        String csv = write("word,amount\nabc,1\n");

        IOException e = assertThrows(IOException.class, () -> encoder.load(csv));

        assertEquals("test.csv: expected the header word,count", e.getMessage());
    }

    @Test
    void emptyFileNamesTheExpectedHeader() throws IOException {
        String csv = write("");

        IOException e = assertThrows(IOException.class, () -> encoder.load(csv));

        assertEquals("test.csv: expected the header word,count", e.getMessage());
    }

    @Test
    void wrongColumnCountNamesTheLine() throws IOException {
        String csv = write("word,count\nabc,1\nde,2,3\n");

        IOException e = assertThrows(IOException.class, () -> encoder.load(csv));

        assertEquals("test.csv, line 3: expected 2 columns, found 3", e.getMessage());
    }

    @Test
    void aBadValueNamesTheLineAndKeepsTheCause() throws IOException {
        String csv = write("word,count\nabc,1\nde,many\n");

        IOException e = assertThrows(IOException.class, () -> encoder.load(csv));

        assertEquals("test.csv, line 3: For input string: \"many\"", e.getMessage());
        assertInstanceOf(NumberFormatException.class, e.getCause());
    }
}
