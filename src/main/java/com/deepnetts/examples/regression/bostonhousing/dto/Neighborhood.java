package com.deepnetts.examples.regression.bostonhousing.dto;

// The input of the network: one neighborhood (a town or district in a city).
public record Neighborhood(
        double crimeRate,                // crimes per person (higher = more crime)
        double avgRooms,                 // average number of rooms per home
        double oldBuildingsPercent,      // share of homes built long ago, 0-100
        double distanceToEmploymentKm,   // distance to the main places of work, in km
        int highwayAccessIndex,          // how easy it is to reach highways, 1-24 (higher = easier)
        double pupilTeacherRatio,        // pupils per teacher in local schools
        double nitrogenOxidesPpm,        // air pollution (nitrogen oxides), 0-1
        double lowerStatusPercent) {     // share of residents with lower social status, 0-100
}
