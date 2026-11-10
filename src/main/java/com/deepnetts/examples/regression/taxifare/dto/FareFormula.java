package com.deepnetts.examples.regression.taxifare.dto;

// A fare formula: fare = start price + price per km * km + price per minute * minutes.
public record FareFormula(
        double startPriceEur,       // what every ride costs before it starts
        double pricePerKmEur,       // what each km adds
        double pricePerMinuteEur) { // what each minute adds
}
