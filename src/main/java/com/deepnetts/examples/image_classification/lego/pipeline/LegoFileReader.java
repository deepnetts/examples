package com.deepnetts.examples.image_classification.lego.pipeline;

import java.awt.image.BufferedImage;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import javax.imageio.ImageIO;

// Reads one image from a file and checks that it is really an image.
public final class LegoFileReader {

    private LegoFileReader() {
    }

    // The error says where the image path came from.
    public static BufferedImage read(Path imagePath, String source) throws IOException {
        if (!Files.exists(imagePath)) {
            throw new FileNotFoundException(source + ": image not found: " + imagePath.toAbsolutePath()
                    + " (run the example from the project root folder)");
        }
        BufferedImage image = ImageIO.read(imagePath.toFile());
        if (image == null) {
            throw new IOException(source + ": " + imagePath.getFileName() + " is not an image file (use jpg or png)");
        }
        return image;
    }
}
