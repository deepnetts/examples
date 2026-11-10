package com.deepnetts.examples.regression.sortingalgorithm;

import com.deepnetts.examples.regression.sortingalgorithm.dto.ArrayToSort;

// This is the array the network ranks the algorithms for.
// You can describe your own array and see which algorithm the network picks:
// 1. Change the values (or try the example at the bottom).
// 2. Run SortingAlgorithm again.
// 3. Look at the output.
public final class YourArray {

    private YourArray() {
    }

    public static ArrayToSort create() {
        // A big array where half of the numbers are already in their place.
        return new ArrayToSort(
                4000,   // numbers in the array (the data has 100 to 5000)
                50);    // percent already in their place (0 = random order, 100 = sorted)

        // Let's try the opposite: a small list that is almost sorted (for example a few new entries added at the end).
        // To try it, comment out the one above and uncomment the line below.
        //return new ArrayToSort(200, 90);
    }
}
