package com.deepnetts.examples.regression.sortingalgorithm.benchmark;

import com.deepnetts.examples.regression.sortingalgorithm.enums.Algorithm;

// The 9 sorting algorithms that SortingBenchmark measures. Each one sorts an int array in place, from small to large.
// They are the textbook versions, written to be easy to read, not tuned for speed.
public final class Sorts {

    private Sorts() {
    }

    public static void sort(Algorithm algorithm, int[] values) {
        switch (algorithm) {
            case BUBBLE    -> bubble(values);
            case SELECTION -> selection(values);
            case INSERTION -> insertion(values);
            case ODD_EVEN  -> oddEven(values);
            case SHELL     -> shell(values);
            case HEAP      -> heap(values);
            case MERGE     -> merge(values);
            case QUICK     -> quick(values, 0, values.length - 1);
            case BUCKET    -> bucket(values);
        }
    }

    // Swaps neighbours that are in the wrong order. Stops early when a pass swaps nothing,
    // so an array that is already sorted takes one pass.
    private static void bubble(int[] a) {
        for (int end = a.length - 1; end > 0; end--) {
            boolean swapped = false;
            for (int i = 0; i < end; i++) {
                if (a[i] > a[i + 1]) {
                    swap(a, i, i + 1);
                    swapped = true;
                }
            }
            if (!swapped) {
                return;
            }
        }
    }

    // Finds the smallest of the rest and puts it next. Does the same work however sorted the array is.
    private static void selection(int[] a) {
        for (int i = 0; i < a.length - 1; i++) {
            int smallest = i;
            for (int j = i + 1; j < a.length; j++) {
                if (a[j] < a[smallest]) {
                    smallest = j;
                }
            }
            swap(a, i, smallest);
        }
    }

    // Moves each value left until it is in place. Fast when only a few values are out of place.
    private static void insertion(int[] a) {
        for (int i = 1; i < a.length; i++) {
            int value = a[i];
            int j = i - 1;
            while (j >= 0 && a[j] > value) {
                a[j + 1] = a[j];
                j--;
            }
            a[j + 1] = value;
        }
    }

    // Like bubble sort, but compares the odd pairs, then the even pairs, until nothing changes.
    private static void oddEven(int[] a) {
        boolean sorted = false;
        while (!sorted) {
            sorted = true;
            for (int start = 0; start <= 1; start++) {
                for (int i = start; i + 1 < a.length; i += 2) {
                    if (a[i] > a[i + 1]) {
                        swap(a, i, i + 1);
                        sorted = false;
                    }
                }
            }
        }
    }

    // Insertion sort on values far apart first (gap), then closer and closer, ending with gap 1.
    private static void shell(int[] a) {
        for (int gap = a.length / 2; gap > 0; gap /= 2) {
            for (int i = gap; i < a.length; i++) {
                int value = a[i];
                int j = i;
                while (j >= gap && a[j - gap] > value) {
                    a[j] = a[j - gap];
                    j -= gap;
                }
                a[j] = value;
            }
        }
    }

    // Builds a heap (the largest value on top), then moves the top to the end, again and again.
    private static void heap(int[] a) {
        for (int i = a.length / 2 - 1; i >= 0; i--) {
            siftDown(a, i, a.length);
        }
        for (int end = a.length - 1; end > 0; end--) {
            swap(a, 0, end);
            siftDown(a, 0, end);
        }
    }

    private static void siftDown(int[] a, int i, int size) {
        while (true) {
            int largest = i;
            int left = 2 * i + 1;
            int right = left + 1;
            if (left < size && a[left] > a[largest]) {
                largest = left;
            }
            if (right < size && a[right] > a[largest]) {
                largest = right;
            }
            if (largest == i) {
                return;
            }
            swap(a, i, largest);
            i = largest;
        }
    }

    // Sorts each half, then merges the two sorted halves.
    private static void merge(int[] a) {
        mergeSort(a, new int[a.length], 0, a.length);
    }

    private static void mergeSort(int[] a, int[] buffer, int from, int to) {
        if (to - from < 2) {
            return;
        }
        int middle = (from + to) >>> 1;
        mergeSort(a, buffer, from, middle);
        mergeSort(a, buffer, middle, to);
        int left = from;
        int right = middle;
        for (int i = from; i < to; i++) {
            if (right >= to || (left < middle && a[left] <= a[right])) {
                buffer[i] = a[left++];
            } else {
                buffer[i] = a[right++];
            }
        }
        System.arraycopy(buffer, from, a, from, to - from);
    }

    // Splits the array around a pivot (the middle value, so a sorted array is not a slow case), then sorts each side.
    private static void quick(int[] a, int low, int high) {
        while (low < high) {
            int pivot = a[(low + high) >>> 1];
            int i = low;
            int j = high;
            while (i <= j) {
                while (a[i] < pivot) {
                    i++;
                }
                while (a[j] > pivot) {
                    j--;
                }
                if (i <= j) {
                    swap(a, i++, j--);
                }
            }
            // Sorts the smaller side first, so the recursion stays shallow.
            if (j - low < high - i) {
                quick(a, low, j);
                low = i;
            } else {
                quick(a, i, high);
                high = j;
            }
        }
    }

    // Puts each value into one of n buckets by its size, sorts each bucket with insertion sort, then joins them.
    // Fast when the values are spread evenly, as they are here.
    private static void bucket(int[] a) {
        if (a.length < 2) {
            return;
        }
        int min = a[0];
        int max = a[0];
        for (int value : a) {
            min = Math.min(min, value);
            max = Math.max(max, value);
        }
        int bucketCount = a.length;
        int[] counts = new int[bucketCount + 1];
        long range = (long) max - min + 1;
        for (int value : a) {
            counts[bucketOf(value, min, range, bucketCount) + 1]++;
        }
        for (int b = 0; b < bucketCount; b++) {
            counts[b + 1] += counts[b];   // counts[b] = where bucket b starts
        }
        int[] sorted = new int[a.length];
        int[] next = counts.clone();
        for (int value : a) {
            sorted[next[bucketOf(value, min, range, bucketCount)]++] = value;
        }
        for (int b = 0; b < bucketCount; b++) {
            insertionRange(sorted, counts[b], counts[b + 1]);
        }
        System.arraycopy(sorted, 0, a, 0, a.length);
    }

    private static int bucketOf(int value, int min, long range, int bucketCount) {
        return (int) (((long) value - min) * bucketCount / range);
    }

    private static void insertionRange(int[] a, int from, int to) {
        for (int i = from + 1; i < to; i++) {
            int value = a[i];
            int j = i - 1;
            while (j >= from && a[j] > value) {
                a[j + 1] = a[j];
                j--;
            }
            a[j + 1] = value;
        }
    }

    private static void swap(int[] a, int i, int j) {
        int value = a[i];
        a[i] = a[j];
        a[j] = value;
    }
}
