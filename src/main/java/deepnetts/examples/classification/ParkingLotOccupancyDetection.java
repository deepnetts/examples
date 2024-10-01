package deepnetts.examples.classification;

import deepnetts.core.DeepNetts;
import deepnetts.data.ImageSet;
import deepnetts.net.ConvolutionalNetwork;
import deepnetts.net.layers.Filters;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.net.train.BackpropagationTrainer;
import deepnetts.util.DeepNettsException;
import deepnetts.util.FileIO;
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
 * https://www.deepnetts.com/quickstart
 * 
 * Step by step tutorial using Deep Nets GUI
 * https://www.deepnetts.com/blog/parking-lot-occupancy-detection-using-deep-learning-in-java.html
 * 
 */
public class ParkingLotOccupancyDetection {
    
    // dimensions of input images
    int imageWidth = 96;
    int imageHeight = 96;

    static final Logger LOG = Logger.getLogger(DeepNetts.class.getName());


    public void run() throws DeepNettsException, IOException {

        // to run this example you need to download the data set from http://cnrpark.it/ and unpack it, and specify directory below
        ImageSet imageSet = new ImageSet(imageWidth, imageHeight, "D:/datasets/Parkiranje/A");
        imageSet.setInvertImages(true); // optional image preprocessing
        //  imageSet.zeroMean();
        LOG.info("Loading images...");        
       // imageSet.loadLabels(new File(labelsFile));
       // imageSet.loadImages(new File(trainingFile));
        imageSet.shuffle();

        ImageSet[] imageSets = imageSet.split(0.60, 0.40);

        LOG.info("Done loading images.");

        // create convolutional neural network
        LOG.info("Creating neural network...");

        ConvolutionalNetwork parkingNet = ConvolutionalNetwork.builder()
                                            .addInputLayer(imageWidth, imageHeight, 3)
                                            .addConvolutionalLayer(6, Filters.ofSize(3), ActivationType.LEAKY_RELU)
                                            .addMaxPoolingLayer(Filters.ofSize(2).stride(2))
                                            .addFullyConnectedLayer(30, ActivationType.LEAKY_RELU)
                                            .addFullyConnectedLayer(10, ActivationType.LEAKY_RELU)
                                            .addOutputLayer(1, ActivationType.SIGMOID)
                                            .lossFunction(LossType.CROSS_ENTROPY)
                                            .randomSeed(123)
                                            .build();

        LOG.info("Done creating network.");
        LOG.info("Training neural network...");

        // set training options and train the network
        BackpropagationTrainer trainer = parkingNet.getTrainer();
        trainer.setStopError(0.03f)
               .setStopEpochs(15)
               .setLearningRate(0.01f);
        trainer.train(imageSets[0]);

        LOG.info("Done training neural network.");

        EvaluationMetrics  testResults =  parkingNet.test(imageSets[1]);
        System.out.println(testResults);        
        
        // save trained network to file
        try {
            FileIO.writeToFile(parkingNet, "legoPeople.net");
        } catch (IOException ex) {
            Logger.getLogger(ParkingLotOccupancyDetection.class.getName()).log(Level.SEVERE, null, ex);
        }


    }


    public static void main(String[] args) {
        try {
            (new ParkingLotOccupancyDetection()).run();
        } catch (DeepNettsException | IOException ex) {
            Logger.getLogger(ParkingLotOccupancyDetection.class.getName()).log(Level.SEVERE, null, ex);
        }


    }
}