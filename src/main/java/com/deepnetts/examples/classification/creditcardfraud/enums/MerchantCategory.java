package com.deepnetts.examples.classification.creditcardfraud.enums;

// The merchant types the network knows.
// _NET = online purchase, _POS = in-store purchase.
// Order matters: add new ones only at the end, then train the network again.
// A saved network gives wrong answers if the order changes.
public enum MerchantCategory {
    ENTERTAINMENT, FOOD_DINING, GAS_TRANSPORT, GROCERY_NET, GROCERY_POS,
    HEALTH_FITNESS, HOME, KIDS_PETS, MISC_NET, MISC_POS,
    PERSONAL_CARE, SHOPPING_NET, SHOPPING_POS, TRAVEL
}
