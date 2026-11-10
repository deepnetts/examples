package com.deepnetts.examples.classification.spamemail.pipeline;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

// The threshold decides from which probability an email counts as spam: it must be a probability, strictly between 0 and 1.
class SpamPredictorTest {

    @Test
    void aThresholdOutsideZeroToOneIsRejected() {
        for (float threshold : new float[] {0f, 1f, -0.1f, 1.5f}) {
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> new SpamPredictor(null, new EmailEncoder(), threshold));

            assertEquals("spamThreshold must be between 0 and 1, was " + threshold, e.getMessage());
        }
    }

    @Test
    void aThresholdInsideZeroToOneIsAccepted() {
        assertDoesNotThrow(() -> new SpamPredictor(null, new EmailEncoder(), 0.5f));
    }
}
