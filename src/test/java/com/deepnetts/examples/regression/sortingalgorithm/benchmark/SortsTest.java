package com.deepnetts.examples.regression.sortingalgorithm.benchmark;

import com.deepnetts.examples.regression.sortingalgorithm.enums.Algorithm;

import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

import java.util.Arrays;
import java.util.Map;
import java.util.Random;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

// Checks that every algorithm SortingBenchmark measures really sorts, also in the corner cases.
// A wrong sort could be fast and would put a wrong time into the dataset.
class SortsTest {

    @TestFactory
    Stream<DynamicTest> everyAlgorithmSortsEveryKindOfArray() {
        Random random = new Random(1);
        Map<String, int[]> arrays = Map.of(
                "empty", new int[0],
                "one value", new int[] {7},
                "two values", new int[] {2, 1},
                "all equal", new int[] {5, 5, 5, 5, 5},
                "negative values", new int[] {3, -1, 0, -7, 2},
                "already sorted", SortingBenchmark.partlySorted(1000, 100, random),
                "half sorted", SortingBenchmark.partlySorted(1000, 50, random),
                "random order", SortingBenchmark.partlySorted(1000, 0, random),
                "reversed", reversed(1000),
                "many duplicates", random.ints(1000, 0, 10).toArray());

        return Stream.of(Algorithm.values()).flatMap(algorithm -> arrays.entrySet().stream().map(entry ->
                dynamicTest(algorithm.displayName() + ", " + entry.getKey(), () -> {
                    int[] expected = entry.getValue().clone();
                    Arrays.sort(expected);
                    int[] actual = entry.getValue().clone();
                    Sorts.sort(algorithm, actual);
                    assertArrayEquals(expected, actual);
                })));
    }

    private static int[] reversed(int size) {
        int[] array = new int[size];
        for (int i = 0; i < size; i++) {
            array[i] = size - i;
        }
        return array;
    }
}
