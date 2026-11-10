package com.deepnetts.examples.regression.bostonhousing.dto;

// The output of the network: the typical home value in a neighborhood.
public record HomeValuePrediction(
        float medianHomeValueKusd) {   // typical home value, in thousands of US dollars (kUSD)
}
