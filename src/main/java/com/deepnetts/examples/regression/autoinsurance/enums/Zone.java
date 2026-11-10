package com.deepnetts.examples.regression.autoinsurance.enums;

// The zones the network knows (the area where the insured vehicles are mostly driven).
// Order matters: if you add new ones at the end, train the network again.
public enum Zone {
    METRO,          // the biggest cities and their surroundings
    CITY_SMALL,     // other large cities
    TOWN,           // smaller cities and towns
    RURAL_SOUTH,
    RURAL_NORTH,
    NORTH_TOWN,     // towns in the far north
    GOTLAND         // the island of Gotland
}
