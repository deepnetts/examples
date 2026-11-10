package com.deepnetts.examples.regression.taxifare.dto;

// The output of the network: the fare it expects for a ride.
public record FarePrediction(
        double fareEur) {   // in EUR
}
