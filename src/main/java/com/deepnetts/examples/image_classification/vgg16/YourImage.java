package com.deepnetts.examples.image_classification.vgg16;

import java.nio.file.Path;

// This is the image VGG16 judges.
// You can use your own photo (jpg or png) and see what VGG16 thinks:
// 1. Change the path (or try the examples at the bottom).
// 2. Run Vgg16Import again.
// 3. Look at the output.
// VGG16 knows 1000 everyday things (animals, vehicles, food, tools...): see datasets/ImageNet/labels.csv.
public final class YourImage {

    private YourImage() {
    }

    public static Path path() {
        // A photo of an elephant.
        return Path.of("datasets/ImageNet/test-images/elephant.jpg");

        // Let's try other photos: a hot dog, and a dish that is not a hot dog.
        // To try one, comment out the line above and uncomment one below.
        //return Path.of("datasets/HotDogSet/hot_dog/1000288.jpg");
        //return Path.of("datasets/HotDogSet/negative/100135.jpg");
    }
}
