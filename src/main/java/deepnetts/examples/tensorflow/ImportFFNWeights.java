package deepnetts.examples.tensorflow;

import deepnetts.util.TensorflowUtils;
import deepnetts.core.DeepNetts;
import deepnetts.data.DataSets;
import deepnetts.data.TrainTestPair;
import deepnetts.data.norm.MaxScaler;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import java.io.IOException;
import javax.visrec.ml.data.DataSet;
import javax.visrec.ml.eval.EvaluationMetrics;

/**
 * Example how to import weights of feed forward neural network trained with tensorflow.
 * This example imports small feed forward neural network trained for iris classification dataset.
 */
public class ImportFFNWeights {
    public static void main(String[] args) throws IOException {
                   
        // step 1: create the network that will import weights
        FeedForwardNetwork network = FeedForwardNetwork.builder()
                                                        .addInputLayer(4)
                                                        .addFullyConnectedLayer(16, ActivationType.RELU)
                                                        .addOutputLayer(3, ActivationType.SOFTMAX)
                                                        .lossFunction(LossType.CROSS_ENTROPY)
                                                        .build();
       
        // step 2; read exported weights and biases from file and set the weights in network above
        TensorflowUtils.importWeights(network, "iris_exported_weights.txt");
        
          
        // step 3: test the network with imported weights with dataset
        DataSet dataSet = DataSets.readCsv("iris-flowers.csv", 4, 3, true, ",");
        TrainTestPair trainTest = DataSets.trainTestSplit(dataSet, 0.65);

        // normalize data using max normalization
        MaxScaler scaler = new MaxScaler(trainTest.getTrainingeSet());
        scaler.apply(trainTest.getTrainingeSet());   
        scaler.apply(trainTest.getTestSet());  
        
         //evaluate network with the test set
        EvaluationMetrics evalResult = network.test(trainTest.getTestSet());  
        System.out.println(evalResult);
        
        // shutdown all threads
        DeepNetts.shutdown();        
    }
     
}
