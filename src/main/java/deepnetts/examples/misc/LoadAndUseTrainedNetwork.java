package deepnetts.examples.misc;

import deepnetts.net.ConvolutionalNetwork;
import deepnetts.util.FileIO;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.visrec.ri.ml.classification.ImageClassifierNetwork;

/**
 * This example shows how to load and create instance of trained network from file.
 * 
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class LoadAndUseTrainedNetwork {
    
    public static void main(String[] args) {

        try {
            // load trained convolutioal network from file
            ConvolutionalNetwork neuralNet =  FileIO.createFromFile("savedNetwork.dnet", ConvolutionalNetwork.class);

            // create image classifier with loaded neural network
            ImageClassifierNetwork imageClassifier = new ImageClassifierNetwork(neuralNet);
            
            // classify image and get probabilities for each category label
            Map<String, Float> results = imageClassifier.classify(Paths.get("someImage.png"));

            System.out.println(results.toString());

        } catch (IOException | ClassNotFoundException ioe) {
            Logger.getLogger(LoadAndUseTrainedNetwork.class.getName()).log(Level.SEVERE, null, ioe);
        }
     
    }    
}
