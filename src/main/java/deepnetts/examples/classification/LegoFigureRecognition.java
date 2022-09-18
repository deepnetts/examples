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
import deepnetts.util.RandomGenerator;
import java.io.File;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.visrec.ml.eval.EvaluationMetrics;

/**
 * Recognition of Lego figure in image.
 * This examples shows how to use convolutional neural network for binary classification of images.
 *
 * Data set contains about 600 examples of lego figures and non-figures in 96x96 pixel images.
 * 
 * For the best performance and accuracy the recommended way to run this example is to use Deep Netts Pro, with Free Development License.
 * https://www.deepnetts.com/download
 *
 * Step-by-step guide for setting up Deep Netts is available at
 * https://www.deepnetts.com/getting-started
 * 
 * @see ConvolutionalNetwork
 * @see ImageSet
 * 
 */
public class LegoFigureRecognition {

    static final Logger LOG = Logger.getLogger(DeepNetts.class.getName());

    public void run() throws DeepNettsException, IOException {
        int imageWidth = 96;
        int imageHeight = 96;

        RandomGenerator.getDefault().initSeed(123);
        
        String trainingFile = "datasets/LegoPeople/train.txt";
        String labelsFile = "datasets/LegoPeople/labels.txt";
    
        ImageSet imageSet = new ImageSet(imageWidth, imageHeight);

        LOG.info("Loading images...");

        imageSet.loadLabels(new File(labelsFile));
        imageSet.loadImages(new File(trainingFile));
        imageSet.setInvertImages(true);
        imageSet.zeroMean();
        imageSet.shuffle();

        ImageSet[] imageSets = imageSet.split(0.60, 0.40);

        LOG.info("Done loading images.");

        // create convolutional neural network
        LOG.info("Creating neural network...");

        ConvolutionalNetwork legoPeopleNet = ConvolutionalNetwork.builder()
                                                .addInputLayer(imageWidth, imageHeight, 3)
                                                .addConvolutionalLayer(12, Filter.ofSize(5), ActivationType.TANH)
                                                .addMaxPoolingLayer(Filter.ofSize(2).stride(2))
                                                .addFullyConnectedLayer(30, ActivationType.TANH)
                                                .addFullyConnectedLayer(10, ActivationType.TANH)
                                                .addOutputLayer(1, ActivationType.SIGMOID)
                                                .lossFunction(LossType.CROSS_ENTROPY)
                                                .randomSeed(123)
                                                .build();

        LOG.info("Done creating network.");
        LOG.info("Training neural network...");

        // train convolutional network
        BackpropagationTrainer trainer = legoPeopleNet.getTrainer();
        trainer.setLearningRate(0.001f)
                .setStopError(0.05f)
                .setStopEpochs(15);
        
        trainer.train(imageSets[0]);


        EvaluationMetrics  pm =  legoPeopleNet.test(imageSets[1]);
        System.out.println(pm);        
        
        // save trained network to file
        try {
            FileIO.writeToFile(legoPeopleNet, "legoPeople.net");
        } catch (IOException ex) {
            Logger.getLogger(LegoFigureRecognition.class.getName()).log(Level.SEVERE, null, ex);
        }

        DeepNetts.shutdown();

    }




    public static void main(String[] args) {
        try {
            (new LegoFigureRecognition()).run();
        } catch (DeepNettsException | IOException ex) {
            Logger.getLogger(LegoFigureRecognition.class.getName()).log(Level.SEVERE, null, ex);
        }


    }
}