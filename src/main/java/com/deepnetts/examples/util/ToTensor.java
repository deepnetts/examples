package com.deepnetts.examples.util;

import deepnetts.tensor.Tensor1D;

// Converts a record into the numbers the network understands (a tensor: Deep Netts' array of numbers).
public interface ToTensor<R extends Record> {
    Tensor1D toTensor(R record);
}
