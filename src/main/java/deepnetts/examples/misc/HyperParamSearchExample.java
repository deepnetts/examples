package deepnetts.examples.misc;

import deepnetts.automl.FeedForwardNetworkFactory;
import deepnetts.automl.HyperParameterSearch;
import static deepnetts.automl.HyperParameterSearch.GRID;
import static deepnetts.automl.Parameters.HIDDEN_NEURONS;
import static deepnetts.automl.Parameters.LEARNING_RATE;
import static deepnetts.automl.Parameters.MAX_EPOCHS;
import static deepnetts.automl.Parameters.HIDDEN_LAYERS;
import static deepnetts.automl.Parameters.OPTIMIZER;
import deepnetts.automl.Range;
import deepnetts.core.DeepNetts;
import deepnetts.data.DataSets;
import deepnetts.eval.ClassifierEvaluator;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import java.io.IOException;
import java.util.Arrays;
import javax.visrec.ml.data.DataSet;

/**
 * Example of hyper parameter search for classification problem.
 */
public class HyperParamSearchExample {

    public static void main(String[] args) throws IOException {

        int numInputs= 29;
        int numOutputs = 1;
        boolean hasHeader = true;        
               
        DataSet dataSet = DataSets.readCsv("datasets/creditcard-balanced.csv", numInputs, numOutputs, hasHeader);
        DataSets.scaleToMax(dataSet);
        DataSet[] trainTest = dataSet.split(0.7, 0.3);
               
        // ovaj factory kreira mreze i ovde mu se odmah zadaju fiksni parametri: hidden activation
        FeedForwardNetworkFactory networkFactory = new FeedForwardNetworkFactory();
        networkFactory.setNumInputs(numInputs);
        networkFactory.setNumOutputs(numOutputs);
        networkFactory.setLossType(LossType.CROSS_ENTROPY);
        networkFactory.setHiddenActivation(ActivationType.TANH);
        networkFactory.setOutputActivation(ActivationType.SIGMOID);
                
        // ovde se zadaju parametri koji se variraju - kreiraju se kombinacije od lista mogucih vrednosti
        HyperParameterSearch paramSearch = new HyperParameterSearch(); // maybe use builder for this        
        paramSearch.paramValues(OPTIMIZER, Arrays.asList("SGD", "MOMENTUM")) // ove ce da proba sve, to je kao grid
                   .paramRange(LEARNING_RATE, Range.of(0.01f, 0.9f).step(0.1f)) //  DIVIDE_AND_CONQUER ovog parametra uopste i nema!!!
                   .paramRange(HIDDEN_LAYERS, Range.of(1, 3).step(1)) // a svaki da bude max sirinedefinisan HIDDEN_NEURONS
                   .paramRange("hiddenLayer_1", Range.of(4, 10).step(2))   // specify range and optional step, neurons in single hidden layer                                    
                   .paramRange("hiddenLayer_2", Range.of(4, 10).step(2))   // specify range and optional step, neurons in single hidden layer                                    
                   .paramRange("hiddenLayer_3", Range.of(4, 10).step(2))   // specify range and optional step, neurons in single hidden layer                                    
                   .paramValue(MAX_EPOCHS, 100)                   
                   .networkFactory(networkFactory) // network architecture should be changed during the training - providue builder, order of layers?
                   .evaluator(new ClassifierEvaluator())
                   .trainingSet(trainTest[0])
                   .testSet(trainTest[1])
                   .randomSeed(1234);
     
        paramSearch.run();    
        
        System.out.println(paramSearch.getResults());
        
        // shutdown the thread pool
        DeepNetts.shutdown();        
    }
}
