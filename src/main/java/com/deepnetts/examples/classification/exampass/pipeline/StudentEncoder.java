package com.deepnetts.examples.classification.exampass.pipeline;

import com.deepnetts.examples.classification.exampass.dto.PassWeights;
import com.deepnetts.examples.classification.exampass.dto.Student;
import com.deepnetts.examples.util.CsvEncoder;

import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.layers.OutputLayer;
import deepnetts.tensor.Tensor1D;
import deepnetts.tensor.Tensor2D;

import java.util.List;

// Turns data into the numbers the network learns from:
// a CSV row becomes a Student (toRecord), a Student becomes the network inputs (toTensor),
// and the label column becomes the correct answer (toTarget). load() (from CsvEncoder) reads the whole CSV this way.
public final class StudentEncoder implements CsvEncoder<Student> {

    // The expected CSV header: the two student fields and the label (passed).
    private static final List<String> HEADER = List.of("studyHours", "practiceTests", "passed");

    // The label is the correct answer the network learns: 1 = passed, 0 = failed.
    private static final int LABEL_COLUMN = HEADER.size() - 1;

    public static final int NUM_INPUTS = FeatureExtractor.NUM_FEATURES;
    public static final int NUM_OUTPUTS = 1;   // one output: the probability to pass

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

    // Converts a student into the numbers the network uses (a tensor: Deep Netts' array of numbers).
    @Override
    public Tensor1D toTensor(Student student) {
        StudentValidator.validate(student);
        return Tensor1D.of(FeatureExtractor.extract(student));
    }

    // Converts a CSV row into a student.
    // CSV columns: studyHours, practiceTests.
    @Override
    public Student toRecord(String[] columns) {
        return new Student(Double.parseDouble(columns[0]), Integer.parseInt(columns[1]));
    }

    // The correct answer of a CSV row: 1 = passed, 0 = failed.
    @Override
    public Tensor1D toTarget(String[] columns) {
        String label = columns[LABEL_COLUMN];
        return switch (label) {
            case "1" -> Tensor1D.of(new float[] {1f});
            case "0" -> Tensor1D.of(new float[] {0f});
            default -> throw new IllegalArgumentException("passed must be 0 or 1, was " + label);
        };
    }

    // Reads what the network learned about each input.
    // A network without a hidden layer adds up: bias + weight1 * input1 + weight2 * input2, and SIGMOID turns the sum
    // into a probability. The inputs were divided by their largest values, so the weights are turned back
    // into "per hour" and "per test" here.
    public PassWeights learnedWeights(FeedForwardNetwork network) {
        OutputLayer output = (OutputLayer) network.getLayers().getLast();
        Tensor2D weights = output.getWeights();
        return new PassWeights(
                output.getBiases().get(0),
                weights.get(0, 0) / FeatureExtractor.MAX_STUDY_HOURS,
                weights.get(0, 1) / FeatureExtractor.MAX_PRACTICE_TESTS);
    }
}
