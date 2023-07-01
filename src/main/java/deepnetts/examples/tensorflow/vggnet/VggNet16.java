package deepnetts.examples.tensorflow.vggnet;

import deepnetts.net.ConvolutionalNetwork;
import deepnetts.util.FileIO;
import deepnetts.util.Tensor;
import java.io.IOException;
import java.util.Map;

/**
 * User-friendly wrapper for large convolutional neural network VggNet16 pre-trained in Keras/Tensorflow 
 * to recognize 1000 different objects on a large-scale dataset imagenet.
 * 
 * Original paper about VGGNet is available at https://arxiv.org/abs/1409.1556
 * 
 */
public final class VggNet16 {
      
    private final ConvolutionalNetwork convNet;

    public VggNet16(ConvolutionalNetwork convNet) {
        this.convNet = convNet;
    }
        
    public synchronized String guessLabel(String imageFile) throws IOException {
        VggNet16InputImage vggInputImage = new VggNet16InputImage(imageFile);          
        return guessLabel(vggInputImage);
    }    
    
    public synchronized String guessLabel(VggNet16InputImage vggInputImage) {
        Tensor prediction = convNet.predict(vggInputImage.getInput());
        int maxIdx = maxIdxOf(prediction);
        
        return convNet.getOutputLabel(maxIdx);
    }
    
    public String[] getLabels() {
        return convNet.getOutputLabels();
    }
    
    public Map<String, Float> getProbabilitiesForLabels() {
        throw  new UnsupportedOperationException();
    }
       
    static int maxIdxOf(Tensor prediction) {
        int maxIdx = -1;
        float max = 0;
        final float[] predictions = prediction.getValues();
        for(int i=0; i<prediction.size(); i++) {
            if (predictions[i] > max) {
                max = predictions[i];
                maxIdx = i;
            }
        } 
        
        return maxIdx;
    }    
    
    public static VggNet16 fromFile(String fileName) throws IOException, ClassNotFoundException {
        ConvolutionalNetwork convNet = FileIO.createFromFile(fileName, ConvolutionalNetwork.class);  
        
        return new VggNet16(convNet);
    }
        
}
