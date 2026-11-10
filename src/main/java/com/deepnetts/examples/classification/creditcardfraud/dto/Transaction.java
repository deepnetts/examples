package com.deepnetts.examples.classification.creditcardfraud.dto;

import com.deepnetts.examples.classification.creditcardfraud.enums.MerchantCategory;

// The input of the network: one credit-card transaction.
public record Transaction(
        double amount,                       // the amount of money spent, 0 or more
        MerchantCategory merchantCategory,   // where the money was spent
        int hourOfDay,                       // 0-23
        double distanceFromHomeKm) {         // distance from the card holder's home, in km
}
