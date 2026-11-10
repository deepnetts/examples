package com.deepnetts.examples.image_classification.digits.pipeline;

import com.deepnetts.examples.image_classification.digits.dto.DigitMistake;

// Counts, for every real digit, which digit the network said.
// With 10 classes there is no single "yes" and "no", so instead of 4 counts we get a 10 x 10 table:
// row = the real digit, column = the network's answer. The diagonal holds the correct answers.
public final class DigitConfusion {

    private final int[][] counts = new int[DigitEncoder.NUM_OUTPUTS][DigitEncoder.NUM_OUTPUTS];

    private DigitConfusion() {
    }

    public static DigitConfusion of(int[] actual, int[] predicted) {
        DigitConfusion confusion = new DigitConfusion();
        for (int i = 0; i < actual.length; i++) {
            confusion.counts[actual[i]][predicted[i]]++;
        }
        return confusion;
    }

    // How many images of the real digit got this answer.
    public int count(int actual, int predicted) {
        return counts[actual][predicted];
    }

    // How many test images show this digit.
    public int actualCount(int actual) {
        int sum = 0;
        for (int count : counts[actual]) {
            sum += count;
        }
        return sum;
    }

    public int total() {
        int sum = 0;
        for (int digit = 0; digit < DigitEncoder.NUM_OUTPUTS; digit++) {
            sum += actualCount(digit);
        }
        return sum;
    }

    public int correct() {
        int sum = 0;
        for (int digit = 0; digit < DigitEncoder.NUM_OUTPUTS; digit++) {
            sum += counts[digit][digit];
        }
        return sum;
    }

    // Always answering with the most common digit would be right this many times.
    public int baselineCorrect() {
        int best = 0;
        for (int digit = 0; digit < DigitEncoder.NUM_OUTPUTS; digit++) {
            best = Math.max(best, actualCount(digit));
        }
        return best;
    }

    // The wrong answer the network gave most often (the first one, if several are tied).
    public DigitMistake mostCommonMistake() {
        DigitMistake worst = new DigitMistake(0, 0, 0);
        for (int actual = 0; actual < DigitEncoder.NUM_OUTPUTS; actual++) {
            for (int predicted = 0; predicted < DigitEncoder.NUM_OUTPUTS; predicted++) {
                if (actual != predicted && counts[actual][predicted] > worst.count()) {
                    worst = new DigitMistake(actual, predicted, counts[actual][predicted]);
                }
            }
        }
        return worst;
    }
}
