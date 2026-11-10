package com.deepnetts.examples.classification.exampass.dto;

// How much each input pushes a student toward passing, read out of a network without a hidden layer.
// The network adds these up (start + perStudyHour * hours + perPracticeTest * tests) and turns the sum into a probability.
public record PassWeights(
        double start,             // the sum for a student who didn't study and did no practice test
        double perStudyHour,      // what each hour of study adds
        double perPracticeTest) { // what each practice test adds
}
