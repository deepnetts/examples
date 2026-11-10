package com.deepnetts.examples.classification.sonarmines.dto;

import com.deepnetts.examples.classification.sonarmines.enums.SeabedType;

// The input of the network: one sonar contact, an object on the seabed that sent back an echo.
public record SonarContact(
        double rangeM,               // distance from the sonar to the object, in meters
        int aspectAngleDeg,          // angle at which the sonar "sees" the object, 0-180
        double echoLevelDb,          // strength of the returned echo, in decibels (dB); closer to 0 = stronger
        double echoDurationMs,       // how long the echo lasts, in milliseconds
        double spectralCentroidKhz,  // the "average pitch" of the echo, in kilohertz
        double shadowLengthM,        // length of the acoustic shadow behind the object, in meters
        SeabedType seabedType) {
}
