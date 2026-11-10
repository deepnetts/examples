package com.deepnetts.examples.image_classification.lego;

import java.nio.file.Path;

// This is the picture the network judges.
// You can use your own picture (jpg or png) and see what the network thinks:
// 1. Change the path (or try the example at the bottom).
// 2. Run LegoFigure again.
// 3. Look at the output.
public final class YourImage {

    private YourImage() {
    }

    public static Path path() {
        // A picture of a Lego figure.
        return Path.of("datasets/LegoSet/legoman/lego_001.jpg");

        // Let's try the opposite: a picture of something else.
        // To try it, comment out the one above and uncomment the line below.
        //return Path.of("datasets/LegoSet/negative/duke_001.jpg");
    }
}
