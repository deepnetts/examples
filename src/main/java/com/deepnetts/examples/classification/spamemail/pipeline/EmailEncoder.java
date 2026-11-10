package com.deepnetts.examples.classification.spamemail.pipeline;

import com.deepnetts.examples.classification.spamemail.dto.Email;
import com.deepnetts.examples.util.ToTensor;

import deepnetts.tensor.Tensor1D;

// Bridge between our records and the network.
public final class EmailEncoder implements ToTensor<Email> {

    public static final int NUM_INPUTS = FeatureExtractor.NUM_FEATURES;
    public static final int NUM_OUTPUTS = 1;   // one output: the spam probability

    // Converts an email into the numbers the network uses (tensors).
    @Override
    public Tensor1D toTensor(Email email) {
        EmailValidator.validate(email);
        return Tensor1D.of(FeatureExtractor.extract(email));
    }
}
