package com.deepnetts.examples.classification.lightswitch.dto;

// The output of the network: how sure it is that the light is on, and the final answer.
public record LightPrediction(
        float onProbability,   // 0 to 1 (closer to 1 = more likely on)
        boolean isOn) {        // final answer: on / off
}
