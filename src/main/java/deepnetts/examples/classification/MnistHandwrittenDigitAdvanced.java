package deepnetts.examples.classification;

import deepnetts.core.DeepNetts;
import deepnetts.data.ImageSet;
import deepnetts.eval.ClassificationMetrics;
import deepnetts.net.ConvolutionalNetwork;
import deepnetts.net.train.BackpropagationTrainer;
import deepnetts.util.DeepNettsException;
import deepnetts.eval.ClassifierEvaluator;
import deepnetts.eval.ConfusionMatrix;
import deepnetts.examples.util.ExampleDataSets;
import javax.visrec.ml.eval.EvaluationMetrics;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.net.train.opt.OptimizerType;
import deepnetts.util.FileIO;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
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
 * https://www.deepnetts.com/quickstart
 * 
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class MnistHandwrittenDigitAdvanced {

    // dimensions of input images
    int imageWidth = 28;
    int imageHeight = 28;

    // training image index and labels
    String labelsFile = "datasets/mnist/labels.txt"; // data set ne sme da bud elokalni - neka ga downloaduuje sa github-a - mozda visrec?
    String trainingFile = "datasets/mnist/train.txt";

    static final Logger LOGGER = Logger.getLogger(DeepNetts.class.getName());

    public void run() throws DeepNettsException, IOException {

        // download MNIST data set from github
        Path mnistPath = ExampleDataSets.downloadMnistDataSet();   
        LOGGER.info("Downloaded MNIST data set to "+mnistPath);        
        
    
        // create a data set from images and labels
        ImageSet imageSet = new ImageSet(imageWidth, imageHeight);
        imageSet.setInvertImages(true);       
        LOGGER.info("Loading images...");         
        imageSet.loadLabels(new File(labelsFile)); // file with category labels, in this case digits 0-9
        imageSet.loadImages(new File(trainingFile), 1000); // files with list of image paths to use for training,  the second parameter is a number of images in subset of original data set

        ImageSet[] imageSets = imageSet.split(0.65, 0.35); // split data set into training and test sets in given ratio
        int labelsCount = imageSet.getLabelsCount(); // the number of image categories/classes, the number of network outputs should correspond to this

        LOGGER.info("Creating neural network...");

        // create convolutional neural network architecture
        ConvolutionalNetwork neuralNet = ConvolutionalNetwork.builder()
                .addInputLayer(imageWidth, imageHeight)
                .addConvolutionalLayer(12, 5)
                .addMaxPoolingLayer(2, 2)
                .addFullyConnectedLayer(30)
                .addOutputLayer(labelsCount, ActivationType.SOFTMAX)
                .hiddenActivationFunction(ActivationType.TANH)
                .lossFunction(LossType.CROSS_ENTROPY)
                .randomSeed(123)
                .build();

        LOGGER.info("Training neural network");

        // set training options and train the network
        BackpropagationTrainer trainer = neuralNet.getTrainer();
        trainer.setLearningRate(0.001f) // za ada delta 0.00001f za rms prop 0.001
               .setStopError(0.03f)
               .setOptimizer(OptimizerType.ADAGRAD) // use adagrad optimization algorithm
               .setLearningRate(0.001f)
               .setStopEpochs(1000);
        trainer.train(imageSets[0]);

        // Test/evaluate trained network to see how it perfroms with enseen data
        ClassifierEvaluator evaluator = new ClassifierEvaluator();
        EvaluationMetrics em = evaluator.evaluate(neuralNet, imageSets[1]);
        LOGGER.info("------------------------------------------------");
        LOGGER.info("Classification performance measure" + System.lineSeparator());
        LOGGER.info("TOTAL AVERAGE");
        LOGGER.info(evaluator.getMacroAverage().toString());
        LOGGER.info("By Class");
        Map<String, ClassificationMetrics> byClass = evaluator.getMetricsByClass();
        byClass.entrySet().stream().forEach((entry) -> {
            LOGGER.info("Class " + entry.getKey() + ":");
            LOGGER.info(entry.getValue().toString());
            LOGGER.info("----------------");
        });

        ConfusionMatrix cm = evaluator.getConfusionMatrix();
        LOGGER.info(cm.toString());
        
        // Save trained network to file
        FileIO.writeToFile(neuralNet, "mnistDemo.dnet");
        
        // shutdown the thread pool
        DeepNetts.shutdown();            
    }

    public static void main(String[] args) throws IOException {
        (new MnistHandwrittenDigitAdvanced()).run();
    }
}
