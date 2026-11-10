package com.deepnetts.examples.classification.exampass.pipeline;

import com.deepnetts.examples.classification.exampass.dto.Student;

import java.util.List;

// Turns a student into the numbers the network uses. Each of these numbers is called a feature.
public final class FeatureExtractor {

    // The largest values in the data. Dividing by them puts every input between 0 and 1,
    // so hours (up to 50) don't count more than tests (up to 10) just because the numbers are bigger.
    static final double MAX_STUDY_HOURS = 50;
    static final double MAX_PRACTICE_TESTS = 10;

    // Name and order of every input. The order must match extract().
    public static final List<String> FEATURE_NAMES = List.of("study_hours", "practice_tests");

    public static final int NUM_FEATURES = FEATURE_NAMES.size();

    private FeatureExtractor() {
    }

    public static float[] extract(Student student) {
        return new float[] {
                (float) (student.studyHours() / MAX_STUDY_HOURS),
                (float) (student.practiceTests() / MAX_PRACTICE_TESTS)
        };
    }
}
