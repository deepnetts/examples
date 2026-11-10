package com.deepnetts.examples.image_classification.vgg16.pipeline;

import deepnetts.data.ExampleImage;
import deepnetts.tensor.Tensor3D;
import deepnetts.util.ImageUtils;

import java.awt.image.BufferedImage;
import java.awt.image.Raster;

// Bridge between images and VGG16.
// An imported network must get its input prepared exactly like the images it was trained on, in TensorFlow:
// the same size, the same color order and the same scale. Otherwise it still answers, but the answers are wrong.
public final class Vgg16Encoder {

    public static final int IMAGE_SIZE = 224;       // VGG16 was trained on 224 x 224 images
    public static final int IMAGE_CHANNELS = 3;     // blue, green and red

    // The average blue, green and red value of the ImageNet images. VGG16 learned from pixels minus these averages.
    private static final float[] IMAGENET_MEAN_BGR = {103.939f, 116.779f, 123.68f};

    // Converts an image into the numbers VGG16 uses (tensor):
    // stretched to 224 x 224, colors in the order blue, green, red, and the ImageNet average subtracted.
    // Unlike the other examples, the values are not put between 0 and 1: VGG16 was trained on values from about -124 to 152.
    public Tensor3D toTensor(BufferedImage image) {
        return new Vgg16Image(ImageUtils.scaleImage(image, IMAGE_SIZE, IMAGE_SIZE)).getInput();
    }

    // Deep Netts reads an image into red, green, blue values from 0 to 1. This replaces that step with VGG16's own.
    private static final class Vgg16Image extends ExampleImage {

        Vgg16Image(BufferedImage image) {
            super(image);
        }

        @Override
        protected void createInputFromPixels(BufferedImage image, int channels) {
            int width = image.getWidth();
            int height = image.getHeight();
            if (image.getType() != BufferedImage.TYPE_INT_ARGB) {
                BufferedImage copy = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
                copy.getGraphics().drawImage(image, 0, 0, null);
                image = copy;
            }
            Raster raster = image.getRaster();
            float[] values = new float[height * width * channels];
            float[] pixel = null;
            int planeSize = width * height;
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    pixel = raster.getPixel(x, y, pixel);   // red, green, blue, alpha
                    int index = x * height + y;
                    values[index]                 = pixel[2] - IMAGENET_MEAN_BGR[0];   // blue
                    values[planeSize + index]     = pixel[1] - IMAGENET_MEAN_BGR[1];   // green
                    values[2 * planeSize + index] = pixel[0] - IMAGENET_MEAN_BGR[2];   // red
                }
            }
            rgbTensor = new Tensor3D(channels, height, width, values);
        }
    }
}
