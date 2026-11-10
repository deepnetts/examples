package com.deepnetts.examples.classification.irisflower.dto;

import com.deepnetts.examples.classification.irisflower.enums.Species;

// One kind of mistake: a real species the network called another species, and how many times.
public record SpeciesMistake(
        Species actual,      // the real species
        Species predicted,   // what the network said
        int count) {         // how many test flowers got this wrong answer
}
