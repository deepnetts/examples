package deepnetts.examples.misc;

import deepnetts.automl.FeedForwardNetworkFactory;
import deepnetts.automl.HyperParameterSearch;
import static deepnetts.automl.Parameters.HIDDEN_LAYERS;
import static deepnetts.automl.Parameters.LEARNING_RATE;
import static deepnetts.automl.Parameters.STOP_EPOCHS;
import deepnetts.automl.Range;
import deepnetts.core.DeepNetts;
import deepnetts.data.DataSets;
import deepnetts.eval.ClassifierEvaluator;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import java.io.IOException;
import java.util.List;
import javax.visrec.ml.data.DataSet;

/**
 * Example of hyper parameter search for classification problem.
 */
public class HiddenLayersSearchExample {

    public static void main(String[] args) throws IOException {

        int numInputs= 29;
        int numOutputs = 1;
        boolean hasHeader = true;        
               
        DataSet dataSet = DataSets.readCsv("datasets/creditcard-balanced.csv", numInputs, numOutputs, hasHeader);
        DataSets.scaleToMax(dataSet);
        DataSet[] trainTest = dataSet.split(0.7, 0.3);
               
        // ovaj kreira mreze i odmah mu se zadaju fiksni parametri (mogu da ih povucem i iz json fajla)
        FeedForwardNetworkFactory networkFactory = new FeedForwardNetworkFactory();
        networkFactory.setNumInputs(numInputs);
        networkFactory.setNumOutputs(numOutputs);
        networkFactory.setLossType(LossType.CROSS_ENTROPY); // vadi iz nekih podesavanja ne hardkodirano
        networkFactory.setHiddenActivation(ActivationType.TANH); // i ovo isto ne hardkodiraj, povuci iz paramSearch?
        networkFactory.setOutputActivation(ActivationType.SIGMOID); // i ovo isto ne hardkodiraj, povuci iz paramSearch?
                
        // ovaj kombinuje sve vrednosti svakog parametra sa svim vrednostima drugih parametara, parametri su nezavisni
        // kako tretirati zavisne parameter kao broj skrivenih lejera i broj neurona u svakom layeru
        HyperParameterSearch paramSearch = new HyperParameterSearch(); // maybe use builder for this        
        paramSearch.paramRange(HIDDEN_LAYERS, Range.of(1, 3).step(1))  
                   .paramRange("hiddenLayer_1", Range.of(4, 10).step(2))
                   .paramRange("hiddenLayer_2", Range.of(4, 8).step(2))
                   .paramRange("hiddenLayer_3", Range.of(3, 7).step(2))   
                   .paramValues(LEARNING_RATE, List.of(0.01, 0.001))  
                   .paramValue(STOP_EPOCHS, 100)          

                   .networkFactory(networkFactory) 
                   .evaluator(new ClassifierEvaluator())
                   .trainingSet(trainTest[0])
                   .testSet(trainTest[1])
                   .randomSeed(1234);
        
        // kako prosledjivati kada imam konvolucionu mrezu, napravi da radi i z anjih

            // ispisi param search space
     
        paramSearch.run();    
        
        System.out.println(paramSearch.getResults());
      
    }
}
/**
 * 4- 10 skrivenih neurona u svakom sloju
 * po 1, 2, 3 skrivenih slojeva
 * 4, 5, 6, 7, 8, 9
 * 1, 2 srivenih slojeva
 * 
 * ovo generise hidden neurons i hidden layers - taj broj leyera sa ovim neuronima
 * 4, 1
 * 4, 2
 * 4, 3
 * 6, 1
 * 6, 2
 * 6, 3
 * 8, 1
 * 8, 2
 * 8, 3
 * 10, 1
 * 10, 2
 * 10, 3
 *
 * na ovo treba staviti fokus - mozda redizajniraj auto ml prem aovome
 * kako to treba da radi:
 * imam max broj slojeva
 * imam opseg za broj neurona za svaki sloj
 * 1 sloj sa po 4-10 neurona
 * 2 sloja sa po 4-10 neurona (dva prethodna)
 * 3 sloja sa po 4-10 neurona
 * 
 * 
 * 
 * bolje bi bilo da se specificira opseg za svaki sloj
 * 
 * sta se salje u nn factory, kako izgleda param
 * 
 * 
 * Kako treba da izgleda struktura podataka
 * Kako treba da izgleda UI
 * 
 * pseudo kod strukture za generisanje kombinacija
 * to je neka trougaona/nepopunjena matrica ili sl.
 * 
 * for layerIdx = 0; layerIdx < maxHiddenLayers; layerIdx++;
 *  int[][] hiddenLayers = new hiddenLayers[layerIdx+1][layerIdWidth];
 *  for layerWidth = widthMin; layerWidth < widthMax        zameniti sa widthIdx
 *      hiddenLayers[layerIdx][widthIdx] = layerWidth           layerWidth = layerWidths[widthIdx]
 * 
 * 
 * treba mi grid search i jedno iterativno dodavanje layera da bi ih poredio 
 */
