package com.deepnetts.examples.classification.irisflower.dto;

import com.deepnetts.examples.classification.irisflower.enums.Species;

// The output of the network for one flower: the probability of every species and the final answer.
public record SpeciesPrediction(
        float[] probabilities,   // one per species, in the order of Species; they add up to 1
        Species species) {       // the species with the highest probability
}
