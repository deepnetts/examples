package com.deepnetts.examples.image_classification.parking;

import java.nio.file.Path;

// This is the picture of a parking spot the network judges.
// You can use your own picture (jpg or png) and see what the network thinks:
// 1. Change the path (or try the example at the bottom).
// 2. Run ParkingSpot again.
// 3. Look at the output.
// The network learned from one camera looking down at a parking lot, so it works best on pictures like those.
public final class YourImage {

    private YourImage() {
    }

    public static Path path() {
        // Spot 30 at 17:00, with a car in it.
        return Path.of("datasets/ParkingSet/busy/20150703_1700_30.jpg");

        // Let's try the opposite: spot 26 at 17:00, free.
        // To try it, comment out the one above and uncomment the line below.
        //return Path.of("datasets/ParkingSet/negative/20150703_1700_26.jpg");
    }
}
