package com.deepnetts.examples.regression.taxifare.dto;

// The input of the network: one taxi ride.
public record Ride(
        double distanceKm,        // how far the ride went, in km
        double durationMinutes) { // how long it took, in minutes
}
