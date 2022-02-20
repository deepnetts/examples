package deepnetts.examples.misc;

import deepnetts.automl.FeedForwardNetworkFactory;
import deepnetts.automl.HyperParameterSearch;
import static deepnetts.automl.HyperParameterSearch.GRID;
import static deepnetts.automl.Parameters.HIDDEN_LAYERS;
import static deepnetts.automl.Parameters.HIDDEN_NEURONS;
import static deepnetts.automl.Parameters.LEARNING_RATE;
import static deepnetts.automl.Parameters.MAX_EPOCHS;
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
               
        // ovaj kreira mreze i odmah mu se zadaju fiksni parametri
        FeedForwardNetworkFactory networkFactory = new FeedForwardNetworkFactory();
        networkFactory.setNumInputs(numInputs);
        networkFactory.setNumOutputs(numOutputs);
        networkFactory.setLossType(LossType.CROSS_ENTROPY);
        networkFactory.setHiddenActivation(ActivationType.TANH);
                
        HyperParameterSearch paramSearch = new HyperParameterSearch(); // maybe use builder for this        
        paramSearch.param(OPTIMIZER, Arrays.asList("SGD", "MOMENTUM")) // ove ce da proba sve, to je kao grid
                   .param(HIDDEN_NEURONS, Range.of(4, 10).step(2), GRID)   // specify range and optional step, neurons in single hidden layer                                    
                   .param(LEARNING_RATE, Range.of(0.01f, 0.9f).step(0.1f), GRID) //  DIVIDE_AND_CONQUER ovog parametra uopste i nema!!!
                   .param(HIDDEN_LAYERS, Range.of(1, 10).step(1)) // a svaki da bude max sirine hidden neurons ili hidden layer width. Alternativa je da ima niz ranges
                   .param(MAX_EPOCHS, 100)                   
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
