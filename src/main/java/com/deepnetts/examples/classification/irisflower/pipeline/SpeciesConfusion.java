package com.deepnetts.examples.classification.irisflower.pipeline;

import com.deepnetts.examples.classification.irisflower.dto.SpeciesMistake;
import com.deepnetts.examples.classification.irisflower.enums.Species;

// Counts, for every real species, which species the network said.
// With 3 classes there is no single "yes" and "no", so instead of 4 counts we get a 3 x 3 table:
// row = the real species, column = the network's answer. The diagonal holds the correct answers.
public final class SpeciesConfusion {

    private static final int SPECIES = Species.values().length;

    private final int[][] counts = new int[SPECIES][SPECIES];

    private SpeciesConfusion() {
    }

    public static SpeciesConfusion of(Species[] actual, Species[] predicted) {
        SpeciesConfusion confusion = new SpeciesConfusion();
        for (int i = 0; i < actual.length; i++) {
            confusion.counts[actual[i].ordinal()][predicted[i].ordinal()]++;
        }
        return confusion;
    }

    // How many flowers of the real species got this answer.
    public int count(Species actual, Species predicted) {
        return counts[actual.ordinal()][predicted.ordinal()];
    }

    // How many test flowers are of this species.
    public int actualCount(Species actual) {
        int sum = 0;
        for (int count : counts[actual.ordinal()]) {
            sum += count;
        }
        return sum;
    }

    public int total() {
        int sum = 0;
        for (Species species : Species.values()) {
            sum += actualCount(species);
        }
        return sum;
    }

    public int correct() {
        int sum = 0;
        for (Species species : Species.values()) {
            sum += count(species, species);
        }
        return sum;
    }

    // Always answering with the most common species would be right this many times.
    public int baselineCorrect() {
        int best = 0;
        for (Species species : Species.values()) {
            best = Math.max(best, actualCount(species));
        }
        return best;
    }

    // The wrong answer the network gave most often (the first one, if several are tied), or null if there were none.
    public SpeciesMistake mostCommonMistake() {
        SpeciesMistake worst = null;
        for (Species actual : Species.values()) {
            for (Species predicted : Species.values()) {
                int count = count(actual, predicted);
                if (actual != predicted && count > 0 && (worst == null || count > worst.count())) {
                    worst = new SpeciesMistake(actual, predicted, count);
                }
            }
        }
        return worst;
    }
}
