package tfcuda;

import deepnetts.util.TensorflowUtils;
import deepnetts.core.DeepNetts;
import deepnetts.data.DataSets;
import deepnetts.data.MLDataItem;
import deepnetts.data.TabularDataSet;
import deepnetts.data.norm.MaxScaler;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.Mode;
import deepnetts.net.layers.AbstractLayer;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.tensor.TensorBase;
import deepnetts.util.DeepNettsThreadPool;
import deepnetts.util.RandomGenerator;
import org.junit.Test;

import java.io.IOException;

import static org.junit.Assert.*;
import org.junit.Before;

/**
 * Example how to import weights of feed forward neural network trained with tensorflow.
 * This example imports small feed forward neural network trained for iris classification dataset.
 */
public class ImportCudaFowardIrisTest {
    
    DeepNettsThreadPool threadPool;
    TabularDataSet<MLDataItem> dataSet ;
    
    @Before
    public void setUp() throws IOException {
        threadPool = new DeepNettsThreadPool(1);
        
        // fiksiraj random
        RandomGenerator.getDefault().initSeed(1);
        // step 3: test the network with imported weights with dataset
        dataSet = DataSets.readCsv("iris-flowers.csv", 4, 3, true, ",");

        // normalize data using max normalization
        MaxScaler scaler = new MaxScaler(dataSet);
        scaler.apply(dataSet);   
        scaler.apply(dataSet);          
    }       
    
    
    @Test
    public void testImportFromTensorflowForward_CPU() throws IOException {
        DeepNetts.getInstance().setUseCuda(false);        
        DeepNetts.getInstance().setMaxThreads(1);
        FeedForwardNetwork network = FeedForwardNetwork.builder()
                                                        .addInputLayer(4)
                                                        .addFullyConnectedLayer(16, ActivationType.RELU)
                                                        .addOutputLayer(3, ActivationType.SOFTMAX)
                                                        .lossFunction(LossType.CROSS_ENTROPY)
                                                        .build();
       
        // step 2; read exported weights and biases from file and set the weights in network above
        TensorflowUtils.importWeights(network, "iris_exported_weights.txt");        
        
        
        // setuj ulaz da vidis sta ces da dobijes
        TensorBase predicted = network.predict(dataSet.get(0).getInput());
        float[] expected = {0.64183146f, 0.2350864f, 0.123082146f};
        assertArrayEquals(expected, predicted.getValues(), 0f); // to je ako mu je ulaz 0
        
        System.out.println("Layer outputs CPU");
        printLayerOutputs(network, dataSet.get(0).getInput());
        
        
        // evaluate entire data set
//        float acc = evalResult.get(EvaluationMetrics.ACCURACY);
//        float expAcc = 0.8461539f;
//       
//        float f1 = evalResult.get(EvaluationMetrics.F1SCORE);
//        float expF1 = 0.91161615f;
//        
//        assertEquals(expAcc, acc, 0f);
//        assertEquals(expF1, f1, 0f);
                
    }
    

    
    @Test
    public void testImportFromTensorflowForward_CUDA() throws IOException {

        DeepNetts.getInstance().setMaxThreads(1);
        DeepNetts.getInstance().setUseCuda(true);
        
        FeedForwardNetwork network = FeedForwardNetwork.builder()
                                                        .addInputLayer(4)
                                                        .addFullyConnectedLayer(16, ActivationType.RELU)
                                                        .addOutputLayer(3, ActivationType.SOFTMAX)
                                                        .lossFunction(LossType.CROSS_ENTROPY)
                                                        .build();
       
        // step 2; read exported weights and biases from file and set the weights in network above
        TensorflowUtils.importWeights(network, "iris_exported_weights.txt");        
     
        network.setMode(Mode.DEBUG);
        // najverovatnije si mu na ulazu sve nule - neki cuda tenzor nije kopiran
        //kao da s im svi ulazi 0
        TensorBase predicted = network.predict(dataSet.get(0).getInput()); // ovo ne vadi tensor nego kopira vrednosti kojih nema
        
        float[] expected = {0.64183146f,0.2350864f,0.123082146f};
        assertArrayEquals(expected, predicted.getValues(), 0f);        
        
        System.out.println("Layer outputs GPU");
        printLayerOutputs(network, dataSet.get(0).getInput());        
        
        // bolje roveri da li radi forward - da li daje ist rezultate kao cpu
        // kopiraj output tenzore
        
         //evaluate network with the test set
      //  EvaluationMetrics evalResult = network.test(trainTest.getTestSet());  
        //System.out.println(evalResult);
        
//        float acc = evalResult.get(EvaluationMetrics.ACCURACY);
//        float expAcc = 0.8461539f;
//       
//        float f1 = evalResult.get(EvaluationMetrics.F1SCORE);
//        float expF1 = 0.91161615f;
//        
//        assertEquals(expAcc, acc, 0f);
//        assertEquals(expF1, f1, 0f); 
            
    }
    
    /*
    @Test
    public void testIrisCudaBackward() {    
        
    }
    */



  
     private void printLayerOutputs(FeedForwardNetwork network, TensorBase input) {
        network.setInput(input);
        for(AbstractLayer layer : network.getLayers()) {
            System.out.println("Layer output: " + layer.getOutputs());
        }
    }
 }