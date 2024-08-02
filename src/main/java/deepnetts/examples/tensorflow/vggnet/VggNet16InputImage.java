package deepnetts.examples.tensorflow.vggnet;

import deepnetts.data.ExampleImage;
import deepnetts.util.ImageUtils;
import deepnetts.tensor.Tensor3D;
import deepnetts.tensor.TensorBase;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;

public class VggNet16InputImage extends ExampleImage {
    
        
    public VggNet16InputImage(String imgFilePath) throws IOException {
        this(ImageIO.read(new File(imgFilePath)));
    }    
  
    public VggNet16InputImage(BufferedImage img) {
        super(ImageUtils.scaleImage(img, 224, 224));
    }
    
    @Override
    protected void createInputFromPixels(BufferedImage image, int channels) {
        float[] imageNetMean = new float[]{103.939f, 116.779f, 123.68f};
        final int width = image.getWidth();
        final int height = image.getHeight();        
        float[] bgrVector = new float[height * width * channels];

        if (image.getType() != BufferedImage.TYPE_INT_ARGB) {
            BufferedImage imageCopy = new BufferedImage(image.getWidth(), image.getHeight(), BufferedImage.TYPE_INT_ARGB);
            imageCopy.getGraphics().drawImage(image, 0, 0, null);
            image = imageCopy;
        }
        
        Raster raster = image.getRaster();
        float[] pixel = null;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                pixel = raster.getPixel(x, y, pixel);
                bgrVector[y * width + x] = pixel[2] - imageNetMean[0];
                bgrVector[width * height + y * width + x] = pixel[1] - imageNetMean[1];
                bgrVector[2 * width * height + y * width + x] = pixel[0] - imageNetMean[2];                       
            }
        }
        
        rgbTensor = new Tensor3D(height, width, channels, bgrVector);
    }
    
}
