package deepnetts.examples.classification;

import deepnetts.core.DeepNetts;
import deepnetts.data.ImageSet;
import deepnetts.net.ConvolutionalNetwork;
import deepnetts.net.layers.Filter;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.net.train.BackpropagationTrainer;
import deepnetts.util.DeepNettsException;
import deepnetts.util.FileIO;
import java.io.File;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.visrec.ml.eval.EvaluationMetrics;

/**
 * Recognize occupied or empty parking lots on image.
 * This example shows how to use convolutional neural network to perform  binary classification of images.
 * 
 * Data set Description
 * CNRPark+EXT is a dataset for visual occupancy detection of parking lots of roughly 150,000 labeled images (patches) 
 * of vacant and occupied parking spaces, built on a parking lot of 164 parking spaces.
 * Original data set URL: http://cnrpark.it/
 * 
 * For the best performance and accuracy the recommended way to run this example is to use Deep Netts Pro, with Free Development License.
 * https://www.deepnetts.com/download
 *
 * Step-by-step guide for setting up Deep Netts is available at
 * https://www.deepnetts.com/getting-started
 * 
 * Step by step tutorial using Deep Nets GUI
 * https://www.deepnetts.com/blog/parking-lot-occupancy-detection-using-deep-learning-in-java.html
 * 
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class ParkingLotOccupacy {
    
    // dimensions of input images
    int imageWidth = 96;
    int imageHeight = 96;

    // to run this example you need to download the data set from http://cnrpark.it/
    String trainingFile = "D:/datasets/Parkiranje/A/index.txt"; // seth local paths to these files
    String labelsFile = "D:/datasets/Parkiranje/A/labels.txt";

    static final Logger LOG = Logger.getLogger(DeepNetts.class.getName());


    public void run() throws DeepNettsException, IOException {
        ImageSet imageSet = new ImageSet(imageWidth, imageHeight);
       // imageSet.setInvertImages(true); // optional image preprocessing
       // imageSet.zeroMean();        
        LOG.info("Loading images...");        
        imageSet.loadLabels(new File(labelsFile));
        imageSet.loadImages(new File(trainingFile));
        imageSet.shuffle();

        ImageSet[] imageSets = imageSet.split(0.60, 0.40);

        LOG.info("Done loading images.");

        // create convolutional neural network
        LOG.info("Creating neural network...");

        ConvolutionalNetwork parkingNet = ConvolutionalNetwork.builder()
                                            .addInputLayer(imageWidth, imageHeight, 3)
                                            .addConvolutionalLayer(12, Filter.size(3), ActivationType.TANH)
                                            .addMaxPoolingLayer(Filter.size(2).stride(2))
                                            .addFullyConnectedLayer(30, ActivationType.TANH)
                                            .addFullyConnectedLayer(10, ActivationType.TANH)
                                            .addOutputLayer(1, ActivationType.SIGMOID)
                                            .lossFunction(LossType.CROSS_ENTROPY)
                                            .randomSeed(123)
                                            .build();

        LOG.info("Done creating network.");
        LOG.info("Training neural network...");

        // set training options and train the network
        BackpropagationTrainer trainer = parkingNet.getTrainer();
        trainer.setMaxError(0.05f)
               .setMaxEpochs(15)
               .setLearningRate(0.001f);
        trainer.train(imageSets[0]);

        LOG.info("Done training neural network.");

        EvaluationMetrics  testResults =  parkingNet.test(imageSets[1]);
        System.out.println(testResults);        
        
        // save trained network to file
        try {
            FileIO.writeToFile(parkingNet, "legoPeople.net");
        } catch (IOException ex) {
            Logger.getLogger(ParkingLotOccupacy.class.getName()).log(Level.SEVERE, null, ex);
        }

        // shutdown the thread pool
        DeepNetts.shutdown();

    }


    public static void main(String[] args) {
        try {
            (new ParkingLotOccupacy()).run();
        } catch (DeepNettsException | IOException ex) {
            Logger.getLogger(ParkingLotOccupacy.class.getName()).log(Level.SEVERE, null, ex);
        }


    }
}