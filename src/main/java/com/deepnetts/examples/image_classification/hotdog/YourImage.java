package com.deepnetts.examples.image_classification.hotdog;

import java.nio.file.Path;

// This is the photo the network judges.
// You can use your own photo (jpg or png) and see what the network thinks:
// 1. Change the path (or try the example at the bottom).
// 2. Run HotDog again.
// 3. Look at the output.
public final class YourImage {

    private YourImage() {
    }

    public static Path path() {
        // A photo of a hot dog.
        return Path.of("datasets/HotDogSet/hot_dog/1000288.jpg");

        // Let's try the opposite: a photo of other food.
        // To try it, comment out the one above and uncomment the line below.
        //return Path.of("datasets/HotDogSet/negative/100135.jpg");
    }
}
