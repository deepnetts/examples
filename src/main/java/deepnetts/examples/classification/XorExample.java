package deepnetts.examples.classification;

import deepnetts.core.DeepNetts;
import deepnetts.examples.util.ExampleDataSets;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.net.train.BackpropagationTrainer;
import deepnetts.util.DeepNettsException;
import javax.visrec.ml.data.DataSet;

/**
 * A minimal example to train neural network to XOR logical function problem.
 * This is to confirm that back-propagation is working, and that it can
 * solve the simplest nonlinear problem.
 * 
 * XOR Function
 * In1 In2 Out
 *  0   0   0
 *  0   1   1
 *  1   0   1
 *  1   1   0
 * 
 * For the best performance and accuracy the recommended way to run this example is to use Deep Netts Pro, with Free Development License.
 * https://www.deepnetts.com/download
 *
 * Step-by-step guide for setting up Deep Netts is available at
 * https://www.deepnetts.com/quickstart
 * 
 * @see FeedForwardNetwork
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class XorExample {

    public static void main(String[] args) throws DeepNettsException {

        // get XOR data set from provided example data sets
        DataSet dataSet = ExampleDataSets.xor();

        // create a neural network using builder
        FeedForwardNetwork neuralNet = FeedForwardNetwork.builder()
                .addInputLayer(2)
                .addFullyConnectedLayer(3, ActivationType.TANH)
                .addOutputLayer(1, ActivationType.SIGMOID)
                .lossFunction(LossType.MEAN_SQUARED_ERROR)
                .build();
        
        // set training algorithm settings
        BackpropagationTrainer trainer = neuralNet.getTrainer();
        trainer.setStopError(0.001f);
        trainer.setLearningRate(0.1f);
        
        // train the neural network
        trainer.train(dataSet);
        
        // use the neural network to predict result for the given input
        float[] out = neuralNet.predict(0, 1);
        System.out.println("Predicted output: " + out[0]);
                   
    }
}
