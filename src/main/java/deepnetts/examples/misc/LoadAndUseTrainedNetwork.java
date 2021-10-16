package deepnetts.examples.misc;

import deepnetts.util.ConvolutionalImageClassifier;
import deepnetts.net.ConvolutionalNetwork;
import deepnetts.util.FileIO;
import java.io.File;
import java.io.IOException;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.visrec.ml.ClassificationException;

/**
 * This example shows how to load and create instance of trained network from file.
 * 
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class LoadAndUseTrainedNetwork {
    
    public static void main(String[] args) throws ClassificationException {

        try {
            ConvolutionalNetwork neuralNet =  FileIO.createFromFile("savedNetwork.dnet", ConvolutionalNetwork.class);

            ConvolutionalImageClassifier imageClassifier = new ConvolutionalImageClassifier(neuralNet);
            Map<String, Float> results = imageClassifier.classify(new File("someImage.png"));
            System.out.println(results.toString());

        } catch (IOException | ClassNotFoundException ioe) {
            Logger.getLogger(LoadAndUseTrainedNetwork.class.getName()).log(Level.SEVERE, null, ioe);
        }
     
    }    
}
