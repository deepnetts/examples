package com.deepnetts.examples.classification.irisflower;

import com.deepnetts.examples.classification.irisflower.dto.Flower;

// This is the flower the network judges.
// You can create your own flower and see what the network thinks:
// 1. Change the values (or try the example at the bottom).
// 2. Run IrisFlower again.
// 3. Look at the output.
public final class YourFlower {

    private YourFlower() {
    }

    public static Flower create() {
        // A typical versicolor: medium petals.
        return new Flower(
                5.9,    // sepal length in cm
                2.8,    // sepal width in cm
                4.3,    // petal length in cm
                1.3);   // petal width in cm

        // Let's try another species: a flower with long, wide petals (like a virginica).
        // To try it, comment out the flower above and uncomment the line below.
        //return new Flower(6.7, 3.0, 5.6, 2.2);
    }
}
