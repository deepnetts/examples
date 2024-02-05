package deepnetts.examples.classification;

import deepnetts.core.DeepNetts;
import deepnetts.data.DataSets;
import deepnetts.data.ImageSet;
import deepnetts.data.TrainTestSplit;
import deepnetts.eval.ClassificationMetrics;
import deepnetts.net.ConvolutionalNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.train.BackpropagationTrainer;
import deepnetts.eval.ConfusionMatrix;
import deepnetts.eval.Evaluators;
import deepnetts.net.layers.Filter;
import deepnetts.net.layers.Filters;
import deepnetts.net.loss.LossType;
import deepnetts.net.train.opt.OptimizerType;
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
import javax.visrec.ri.ml.classification.ImageClassifierNetwork;

/**
 * Duke Java mascot image recognition.
 * This examples shows how to use convolutional neural network for binary classification of images.
 * This example uses some additional advanced settings for network training.
 *
 * Data set contains 114 images of Duke and Non-Duke examples.
 * 
 * For the best performance and accuracy the recommended way to run this example is to use Deep Netts Pro, with Free Development License.
 * https://www.deepnetts.com/download
 *
 * Step-by-step guide for setting up Deep Netts is available at
 * https://www.deepnetts.com/quickstart
 * 
 * @see ConvolutionalNetwork
 * @see ImageSet
 * @see BackpropagationTrainer
 * @see OptimizerType
 * 
 */
public class DukeDetectorAdvanced {

    static final Logger LOGGER = Logger.getLogger(DeepNetts.class.getName());

    public static void main(String[] args) throws FileNotFoundException, IOException, ClassNotFoundException {
        int imageWidth = 64;
        int imageHeight = 64;

        String dataSetPath = "datasets/DukeSet";        
        String trainingFile = dataSetPath +"/index.txt";
        String labelsFile = dataSetPath +"/labels.txt"; // labels file should be generated uatomaticaly if not present based on the class dir names

        RandomGenerator.getDefault().initSeed(123);
        
        ImageSet imageSet = new ImageSet(imageWidth, imageHeight);
        imageSet.setResizeStrategy(ImageResize.STRATCH);
        imageSet.setInvertImages(true);
        imageSet.zeroMean();        

        LOGGER.info("Loading images...");
        imageSet.loadLabels(new File(labelsFile));
        imageSet.loadImages(new File(trainingFile)); // point to Path

        TrainTestSplit trainTestPair = DataSets.trainTestSplit(imageSet, 0.7) ;

        LOGGER.info("Creating a neural network...");

        ConvolutionalNetwork convNet = ConvolutionalNetwork.builder()
                .addInputLayer(imageWidth, imageHeight, 3)
                .addConvolutionalLayer(32, Filters.ofSize(3, 3), ActivationType.RELU)
                .addMaxPoolingLayer(2, 2, 2)
                .addFullyConnectedLayer(16, ActivationType.RELU)
                .addOutputLayer(1, ActivationType.SIGMOID)
                .lossFunction(LossType.CROSS_ENTROPY)
                .build();

        LOGGER.info("Training the neural network...");

        // Get a trainer of the created convolutional network
        BackpropagationTrainer trainer = convNet.getTrainer();
        trainer.setStopError(0.03f)
               .setOptimizer(OptimizerType.ADAGRAD) // za ada delta skace jer bude preveliki initial learning rate
               .setLearningRate(0.001f);
        trainer.train(trainTestPair.getTrainingSet());

        LOGGER.info("Saving the trained neural network.");
        // save trained neural network to file
        FileIO.writeToFile(convNet, "DukeDetector.dnet");

        LOGGER.info("Test the trained neural network.");
        ClassificationMetrics cem = Evaluators.evaluateClassifier(convNet, trainTestPair.getTestSet()); // vrati evaluatora i iz njega uzmi sve sto ti ytreba
        System.out.println(cem);
        
        ConfusionMatrix confusionMatrix = cem.getConfusionMatrix();
        System.out.println(confusionMatrix);

        // how to use recognizer for single image
        BufferedImage image = ImageIO.read(new File("D:\\datasets\\DukeSet\\duke\\duke7.jpg")); // promeni ovu sliku i ubaci u resources!
        ImageClassifier<BufferedImage> imageClassifier = new ImageClassifierNetwork(convNet);
        Map<String, Float> results = imageClassifier.classify(image);

        System.out.println(results.toString());
        
        // shutdown the thread pool
        DeepNetts.shutdown();
    }

}
