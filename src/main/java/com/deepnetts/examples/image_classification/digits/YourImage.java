package com.deepnetts.examples.image_classification.digits;

import java.nio.file.Path;

// This is the image the network reads.
// You can use your own image (jpg or png) and see what the network thinks:
// 1. Change the path (or try the example at the bottom).
// 2. Run Digits again.
// 3. Look at the output.
// Like the training images, your digit should be light on a dark background, centered and filling most of the image.
public final class YourImage {

    private YourImage() {
    }

    public static Path path() {
        // A handwritten 7.
        return Path.of("datasets/DigitSet/7/02102.png");

        // Let's try another digit: a handwritten 3.
        // To try it, comment out the one above and uncomment the line below.
        //return Path.of("datasets/DigitSet/3/00486.png");
    }
}
