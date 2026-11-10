package com.deepnetts.examples.regression.autoinsurance.dto;

// The output of the network: the expected insurance payment per insured year.
public record PaymentPrediction(
        float paymentPerInsuredYearKsek) {   // expected payment per insured year, in thousands of Swedish kronor (kSEK)
}
