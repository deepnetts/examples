package com.deepnetts.examples.classification.exampass.dto;

// The output of the network: how likely the student is to pass, and the final answer.
public record PassPrediction(
        float passProbability,   // 0 to 1 (closer to 1 = more likely to pass)
        boolean willPass) {      // final answer: pass / fail
}
