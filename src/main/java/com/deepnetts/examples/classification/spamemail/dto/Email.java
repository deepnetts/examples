package com.deepnetts.examples.classification.spamemail.dto;

// The input of the network: one email, described by simple word and character statistics.
// "Percent" fields = how much of the whole email is that word or character (0-100).
public record Email(
        double freeWordPercent,          // how much of the email is the word "free"
        double moneyWordPercent,         // ... the word "money"
        double removeWordPercent,        // ... the word "remove" (as in "click here to remove yourself")
        double yourWordPercent,          // ... the word "your"
        double exclamationCharPercent,   // how much of the email is "!" characters
        double dollarCharPercent,        // how much of the email is "$" characters
        int longestCapitalRun,           // longest run of CAPITAL letters in a row
        int totalCapitalLetters) {       // total number of capital letters
}
