package com.deepnetts.examples.image_classification.dukedetection;

import java.nio.file.Path;

// This is the image the network judges.
// You can use your own image (jpg or png) and see what the network thinks:
// 1. Change the path (or try the example at the bottom).
// 2. Run DukeDetection again.
// 3. Look at the output.
public final class YourImage {

    private YourImage() {
    }

    public static Path path() {
        // An image of Duke.
        return Path.of("datasets/DukeSet/duke/duke4.jpg");

        // Let's try the opposite: a Lego figure, also a figure on a white background, but not Duke.
        // To try it, comment out the one above and uncomment the line below.
        //return Path.of("datasets/DukeSet/negative/lego_01.jpg");
    }
}
