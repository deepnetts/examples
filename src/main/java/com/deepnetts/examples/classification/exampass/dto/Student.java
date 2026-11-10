package com.deepnetts.examples.classification.exampass.dto;

// The input of the network: one student before the exam.
public record Student(
        double studyHours,     // hours spent studying for the exam, 0 or more
        int practiceTests) {   // practice tests done before the exam, 0 or more
}
