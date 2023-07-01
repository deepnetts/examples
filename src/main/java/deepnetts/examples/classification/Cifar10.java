package deepnetts.examples.classification;

import deepnetts.core.DeepNetts;
import deepnetts.data.ImageSet;
import deepnetts.data.MLDataItem;
import deepnetts.eval.ClassificationMetrics;
import deepnetts.net.ConvolutionalNetwork;
import deepnetts.net.train.BackpropagationTrainer;
import deepnetts.util.DeepNettsException;
import deepnetts.eval.ClassifierEvaluator;
import deepnetts.eval.ConfusionMatrix;
import deepnetts.net.layers.Filter;
import javax.visrec.ml.eval.EvaluationMetrics;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.net.train.opt.OptimizerType;
import deepnetts.util.FileIO;
import java.io.File;
import java.io.IOException;
import java.util.Map;
import java.util.logging.Logger;


/**
 * Recognition of hand-written digits.
 * This example shows how to use convolutional neural network to recognize hand written digits.
 * The problem of hand-written digit recognition is solved as multi-class classification of images 
 * 
 * Data set description
 * The data set used in this examples is a subset of original MNIST data set, which is considered to be a 'Hello world' for image recognition.
 * The original data set contains 60000 images,  dimensions 28x28 pixels.
 * Original data set URL: http://yann.lecun.com/exdb/mnist/
 * 
 * For the best performance and accuracy the recommended way to run this example is to use Deep Netts Pro, with Free Development License.
 * https://www.deepnetts.com/download
 *
 * Step-by-step guide for setting up Deep Netts is available at
 * https://www.deepnetts.com/getting-started
 * 
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class Cifar10 {

    // dimensions of input images
    int imageWidth = 32;
    int imageHeight = 32;

    // training image index and labels
    String labelsFile = "d:/datasets/cifar10/train/labels.txt"; // data set ne sme da bud elokalni - neka ga downloaduuje sa github-a - mozda visrec?
    String trainingFile = "d:/datasets/cifar10/train/index.txt";

    static final Logger LOGGER = Logger.getLogger(DeepNetts.class.getName());

    public void run() throws DeepNettsException, IOException {
        
        // create a data set from images and labels
        ImageSet imageSet = new ImageSet(imageWidth, imageHeight);
        imageSet.zeroMean();
      //  imageSet.setInvertImages(true);        
        LOGGER.info("Loading images...");
        imageSet.loadLabels(new File(labelsFile)); // file with category labels, in this case digits 0-9
        imageSet.loadImages(new File(trainingFile)); // 10000 files with list of image paths to use for training,  the second parameter is a number of images in subset of original data set

        ImageSet[] imageSets = imageSet.split(0.65, 0.35); // split data set into training and test sets in given ratio
        int labelsCount = imageSet.getLabelsCount(); // the number of image categories/classes, the number of network outputs should correspond to this

        LOGGER.info("Creating neural network architecture...");

        // create convolutional neural network architecture
        ConvolutionalNetwork neuralNet = ConvolutionalNetwork.builder()
                .addInputLayer(imageWidth, imageHeight)
                .addConvolutionalLayer(3, Filter.ofSize(3))
                .addMaxPoolingLayer(2, 2)     //16
                .addConvolutionalLayer(12, Filter.ofSize(3))
                .addMaxPoolingLayer(2, 2)   // 8
                .addConvolutionalLayer(24, Filter.ofSize(3))
                .addMaxPoolingLayer(2, 2)     //4           
//                .addConvolutionalLayer(24, 5)
//                .addMaxPoolingLayer(2, 2)     // 2
//                .addConvolutionalLayer(48, 5)
//                .addMaxPoolingLayer(2, 2)     // 1                
                .addFullyConnectedLayer(100)
                .addFullyConnectedLayer(100)
                .addOutputLayer(labelsCount, ActivationType.SOFTMAX)
                .hiddenActivationFunction(ActivationType.RELU)
                .lossFunction(LossType.CROSS_ENTROPY)
                .randomSeed(123)
                .build();

        LOGGER.info("Training the neural network");

        // set training options and train the network
        BackpropagationTrainer trainer = neuralNet.getTrainer();
        trainer.setLearningRate(0.001f)
                .setStopError(0.09f)
                .setStopAccuracy(0.95f)
                .setOptimizer(OptimizerType.MOMENTUM)
                .setMomentum(0.7f);
        trainer.train(imageSets[0]);

        // ovde ispisi greske za svaki element data seta
        for(MLDataItem dataItem : imageSets[0]) {
            System.out.println(dataItem.getError());
        }
        
        // Test/evaluate trained network to see how it perfroms with enseen data
        ClassifierEvaluator evaluator = new ClassifierEvaluator();
        EvaluationMetrics em = evaluator.evaluate(neuralNet, imageSets[1]);
        LOGGER.info("------------------------------------------------");
        LOGGER.info("Classification metrics\n");
        LOGGER.info(evaluator.getMacroAverage().toString()); // average metrics for all classes
        LOGGER.info("By Class"); // print evaluation metrics for each class/category
        Map<String, ClassificationMetrics> byClass = evaluator.getMetricsByClass();
        byClass.entrySet().stream().forEach((entry) -> {
            LOGGER.info("Class " + entry.getKey() + ":");
            LOGGER.info(entry.getValue().toString());
            LOGGER.info("----------------");
        });

        ConfusionMatrix confMatrix = evaluator.getConfusionMatrix();
        LOGGER.info(confMatrix.toString());
        
        // Save trained network to file
        FileIO.writeToFile(neuralNet, "cifar10net.dnet");
        
        // shutdown the thread pool
        DeepNetts.shutdown();             
    }

    public static void main(String[] args) throws IOException {
        (new Cifar10()).run();
    }
}
