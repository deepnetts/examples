package com.deepnetts.examples.classification.lightswitch.dto;

// The input of the network: the two switches of a staircase light.
public record Switches(
        boolean bottomUp,   // the switch at the bottom of the stairs is up
        boolean topUp) {    // the switch at the top of the stairs is up
}
