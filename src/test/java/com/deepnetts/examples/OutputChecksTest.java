package com.deepnetts.examples;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

// The check that every example beats its baseline, on lines as the examples print them.
class OutputChecksTest {

    @Test
    void anAccuracyAboveTheBaselinePasses() {
        String output = "  Accuracy  89.6%  - 679 of 758 correct\n  Baseline  56.7%  - always guessing the more common answer would be this accurate\n";

        assertEquals("", OutputChecks.problemWithBaseline(output));
    }

    @Test
    void anAccuracyBelowTheBaselineIsReported() {
        String output = "  Accuracy    47.0 %  - share of images\n  Baseline    52.7%  - always answering\n";

        assertEquals("accuracy 47.0% is not above the baseline 52.7%", OutputChecks.problemWithBaseline(output));
    }

    @Test
    void aSmallerErrorThanTheBaselinePasses() {
        String output = "  MAE       2.5 kUSD  - on average\n  Baseline  7.2 kUSD  - always answering with the average\n";

        assertEquals("", OutputChecks.problemWithBaseline(output));
    }

    @Test
    void theAverageErrorOfTheTaxiExampleIsComparedToo() {
        String output = "  Average error   0.63 EUR  - off by this much\n  Baseline       13.63 EUR  - always answering\n";

        assertEquals("", OutputChecks.problemWithBaseline(output));
    }

    @Test
    void aLargerErrorThanTheBaselineIsReported() {
        String output = "  MAE       9.0 kUSD  - on average\n  Baseline  7.2 kUSD  - always answering with the average\n";

        assertTrue(OutputChecks.problemWithBaseline(output).contains("not below the baseline"));
    }

    @Test
    void theChoiceOfTheSortingExampleIsComparedWithAlwaysTheSameAlgorithm() {
        String good = "  Picked the fastest   64 of 100  - the network's fastest\n  Baseline             46 of 100  - always Quick Sort\n";
        String bad = "  Picked the fastest   40 of 100  - the network's fastest\n  Baseline             46 of 100  - always Quick Sort\n";

        assertEquals("", OutputChecks.problemWithBaseline(good));
        assertTrue(OutputChecks.problemWithBaseline(bad).contains("not more than the baseline 46"));
    }

    @Test
    void anOutputWithoutABaselineIsNotChecked() {
        assertEquals("", OutputChecks.problemWithBaseline("  37 C is 98.6 F, the network says 98.6 F\n"));
    }
}
