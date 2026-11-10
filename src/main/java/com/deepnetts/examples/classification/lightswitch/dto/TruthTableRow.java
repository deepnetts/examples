package com.deepnetts.examples.classification.lightswitch.dto;

// One position of the two switches: the real light, and what each network says about it.
public record TruthTableRow(
        boolean bottomUp,             // the switch at the bottom is up
        boolean topUp,                // the switch at the top is up
        boolean lightOn,              // the real answer from the data
        float withoutHiddenLayer,     // the on-probability of the network without a hidden layer
        float withHiddenLayer) {      // the on-probability of the network with a hidden layer
}
