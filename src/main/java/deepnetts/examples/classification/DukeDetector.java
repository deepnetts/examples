package deepnetts.examples.classification;

import deepnetts.core.DeepNetts;
import deepnetts.data.DataSets;
import deepnetts.data.ImageSet;
import deepnetts.data.TrainTestPair;
import deepnetts.eval.ClassificationMetrics;
import deepnetts.net.ConvolutionalNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.train.BackpropagationTrainer;
import deepnetts.eval.ConfusionMatrix;
import deepnetts.net.layers.Filter;
import deepnetts.net.loss.LossType;
import deepnetts.util.FileIO;
import deepnetts.util.ImageResize;
import deepnetts.util.RandomGenerator;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Map;
import java.util.logging.Logger;
import javax.imageio.ImageIO;
import javax.visrec.ml.classification.ImageClassifier;
import javax.visrec.ml.eval.EvaluationMetrics;
import javax.visrec.ri.ml.classification.ImageClassifierNetwork;

/**
 * Duke Java mascot image recognition.
 * This examples shows how to use convolutional neural network for binary classification of images.
 *
 * Data set contains 114 images of Duke and Non-Duke examples.
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
public class DukeDetector {

    static final Logger LOGGER = Logger.getLogger(DeepNetts.class.getName());

    public static void main(String[] args) throws FileNotFoundException, IOException, ClassNotFoundException {

        // width and height of input image
        int imageWidth = 64;
        int imageHeight = 64;
        
        // paths to training images
        String dataSetPath = "datasets/DukeSet"; // path to folder which contain category sub fodlers with images
        String trainingFile = dataSetPath + "/index.txt"; // path to plain txt file with list of images and coresponding labels
        String labelsFile = dataSetPath + "/labels.txt"; // path to plain file with list of labels

        RandomGenerator.getDefault().initSeed(123); // fix random generator to get repeatable training
        DeepNetts.getInstance();
        // initialize image data set and preprocessing
        LOGGER.info("Loading images...");
        ImageSet imageSet = new ImageSet(imageWidth, imageHeight);
        imageSet.setResizeStrategy(ImageResize.STRATCH);
        imageSet.setInvertImages(true);
        imageSet.zeroMean();
        imageSet.loadLabels(new File(labelsFile));
        imageSet.loadImages(new File(trainingFile)); // point to Path

        // split data into training and test set
        TrainTestPair trainTestPair = DataSets.trainTestSplit(imageSet, 0.7);

        // create a convolutional neural network arhitecture for binary image classification
        LOGGER.info("Creating a neural network...");
        ConvolutionalNetwork convNet = ConvolutionalNetwork.builder()
                .addInputLayer(imageWidth, imageHeight, 3)
                .addConvolutionalLayer(6, Filter.ofSize(3).stride(2), ActivationType.LEAKY_RELU)
                .addMaxPoolingLayer(Filter.ofSize(2).stride(2))
                .addFullyConnectedLayer(16, ActivationType.LEAKY_RELU)
                .addOutputLayer(1, ActivationType.SIGMOID)
                .lossFunction(LossType.CROSS_ENTROPY)
                .build();

        // set training options and run training
        LOGGER.info("Training the neural network...");        
        BackpropagationTrainer trainer = convNet.getTrainer(); // Get a trainer of the created convolutional network
        trainer.setStopError(0.03f)  
               .setLearningRate(0.01f); 
        trainer.train(trainTestPair.getTrainingeSet()); // run training

        LOGGER.info("Test the trained neural network.");
        EvaluationMetrics testResults = convNet.test(trainTestPair.getTestSet());
        System.out.println(testResults);
            
        // get confusion matrix which contains details about correct and wrong classifications        
        ConfusionMatrix confusionMatrix = ((ClassificationMetrics)testResults).getConfusionMatrix(); 
        System.out.println(confusionMatrix);
        
        // save trained neural network to file
        LOGGER.info("Saving the trained neural network.");
        FileIO.writeToFile(convNet, "DukeDetector.dnet");      

        // how to use recognizer for single image
        LOGGER.info("Recognizing an example duke image.");
        
        BufferedImage image = ImageIO.read(new File("datasets/DukeSet/duke/duke7.jpg"));
        ImageClassifier<BufferedImage> imageClassifier = new ImageClassifierNetwork(convNet);
        Map<String, Float> results = imageClassifier.classify(image); // result is a map with image labels as keys and coresponding probability
        LOGGER.info(results.toString());

        // shutdown the thread pool
        DeepNetts.shutdown();
    }

}