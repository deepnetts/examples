package com.deepnetts.examples.util;

// Takes an input and returns a prediction.
public interface Predictor<I, O> {
    O predict(I input);
}
