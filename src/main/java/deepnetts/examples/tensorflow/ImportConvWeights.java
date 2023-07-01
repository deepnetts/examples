package deepnetts.examples.tensorflow;

import deepnetts.util.TensorflowUtils;
import deepnetts.core.DeepNetts;
import deepnetts.data.ImageSet;
import deepnetts.examples.util.ExampleDataSets;
import deepnetts.net.ConvolutionalNetwork;
import deepnetts.net.layers.Filter;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import javax.visrec.ml.eval.EvaluationMetrics;

/**
 * Example how to import weights of convolutional  neural network trained with tensorflow.
 * This example imports small convolutional neural network trained for mnist dataset.
 */
public class ImportConvWeights {
    
    public static void main(String[] args) throws IOException {

        // step 1: create the network that will import weights
        ConvolutionalNetwork network = ConvolutionalNetwork.builder()
                .addInputLayer(28, 28, 1)
                .addConvolutionalLayer(32, Filter.ofSize(3), ActivationType.RELU)
                .addMaxPoolingLayer(2, 2)
                .addFullyConnectedLayer(100, ActivationType.RELU)
                .addOutputLayer(10, ActivationType.SOFTMAX)
                .lossFunction(LossType.CROSS_ENTROPY)
                .build();

        // step 2; read exported weights and biases from file and set the weights in network above
        TensorflowUtils.importWeights(network, "mnist_exported_weights.txt");
        
        // step 3: save network to file (optionally)
        network.save("deepnettsWithImportedWeights.dnet");
        
        // step 4: testing the network with imported weights with dataset          
        // download MNIST data set from github
        Path mnistPath = ExampleDataSets.downloadMnistDataSet();   
        // create a data set from images and labels
        ImageSet imageSet = new ImageSet(28, 28);
        imageSet.setGrayscale(true);
        imageSet.setInvertImages(true);
        imageSet.loadLabels(new File("datasets/mnist/training/labels.txt"));
        imageSet.loadImages(new File("datasets/mnist/training/train.txt"), 1000);

        // test the network using loaded data set
        EvaluationMetrics evalResult = network.test(imageSet);  
        System.out.println(evalResult);
        
        // shutdown all threads
        DeepNetts.shutdown();
    }

}