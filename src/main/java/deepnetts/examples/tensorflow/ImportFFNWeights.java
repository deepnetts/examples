package deepnetts.examples.tensorflow;

import deepnetts.util.TensorflowUtils;
import deepnetts.core.DeepNetts;
import deepnetts.data.DataSets;
import deepnetts.data.MLDataItem;
import deepnetts.data.TabularDataSet;
import deepnetts.data.TrainTestSplit;
import deepnetts.data.norm.MaxScaler;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.Mode;
import deepnetts.net.layers.AbstractLayer;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.tensor.Tensor1D;
import deepnetts.tensor.TensorBase;
import java.io.IOException;
import javax.visrec.ml.eval.EvaluationMetrics;

/**
 * Example how to import weights of feed forward neural network trained with tensorflow.
 * This example imports small feed forward neural network trained for iris classification dataset.
 */
public class ImportFFNWeights {
    public static void main(String[] args) throws IOException {
      //  DeepNetts.getInstance().setMaxThreads(1);// ovo da moze pojedinacne mreze ne sve
        DeepNetts.getInstance().setUseCuda(true);// ovo da moze pojedinacne mreze ne sve

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
        TabularDataSet<?> dataSet = DataSets.readCsv("iris-flowers.csv", 4, 3, true, ",");
        TrainTestSplit trainTest = DataSets.trainTestSplit(dataSet, 0.65);

        // normalize data using max normalization
        MaxScaler scaler = new MaxScaler(trainTest.getTrainingSet());
        scaler.apply(trainTest.getTrainingSet());   
        scaler.apply(trainTest.getTestSet());  
           
        network.setMode(Mode.DEBUG);
         //evaluate network with the test set
        EvaluationMetrics evalResult = network.test(trainTest.getTestSet());  
        System.out.println(evalResult);
        
        testLayerOutputs(network, ((MLDataItem)trainTest.getTestSet().get(0)).getInput());
            
    }
         



  
     static void testLayerOutputs(FeedForwardNetwork network, TensorBase input) {
        network.setInput(input);
        for(AbstractLayer layer : network.getLayers()) {
            System.out.println("Layer output: " + layer.getOutputs());
        }
    }
 }