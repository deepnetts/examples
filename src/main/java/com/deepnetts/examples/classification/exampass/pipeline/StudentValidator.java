package com.deepnetts.examples.classification.exampass.pipeline;

import com.deepnetts.examples.classification.exampass.dto.Student;

// Checks that a student has valid values.
public final class StudentValidator {

    private StudentValidator() {
    }

    // Same check, but the error says where the student came from.
    public static void validate(Student student, String source) {
        try {
            validate(student);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(source + ": " + e.getMessage(), e);
        }
    }

    // Throws an error naming the broken rule and the bad value.
    public static void validate(Student student) {
        if (student == null) {
            throw new IllegalArgumentException("student must not be null");
        }
        if (!Double.isFinite(student.studyHours()) || student.studyHours() < 0) {
            reject("studyHours must be a number >= 0", student.studyHours());
        }
        if (student.practiceTests() < 0) {
            reject("practiceTests must be 0 or more", student.practiceTests());
        }
    }

    // Throws an error naming the broken rule and the bad value: "<rule>, was <value>".
    private static void reject(String rule, Object actualValue) {
        throw new IllegalArgumentException(rule + ", was " + actualValue);
    }
}
