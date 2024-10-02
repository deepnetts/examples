package deepnetts.examples.classification;

import deepnetts.core.DeepNetts;
import deepnetts.data.DataSets;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.util.DeepNettsException;
import java.io.IOException;
import javax.visrec.ml.data.DataSet;
import javax.visrec.ml.eval.EvaluationMetrics;
import javax.visrec.ml.classification.BinaryClassifier;
import javax.visrec.ri.ml.classification.FeedForwardNetBinaryClassifier;
import deepnetts.data.MLDataItem;

/**
 * Credit Card Fraud Detection. 
 * This example demonstrates how to use Feed Forward Neural Network to perform binary classification task.
 * The trained network classifies credit card transactions into one of two possible categories: fraud / not fraud (true/false)
 * The output of the network is probability that a given transaction is fraud.
 *
 * Data Set description.
 * The data set contains transactions made by credit cards in September 2013 by European cardholders.
 * This data set presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions.
 * The data set is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
 * For the purposes of this example data set has been balanced by using a subset of the original data set.
 * All attributes in data set are  anonymized except last two which represent transaction amount and class. 
 * URL: https://www.kaggle.com/mlg-ulb/creditcardfraud
 * 
 * For the best performance and accuracy the recommended way to run this example is to use Deep Netts Pro, with Free Development License.
 * https://www.deepnetts.com/download
 *
 * Step-by-step guide for setting up Deep Netts is available at
 * https://www.deepnetts.com/quickstart
 * 
 * @see FeedForwardNetwork
 * @see BinaryClassifier
 * 
 */
public class CrediCardFraud {

    public static void main(String[] args) throws DeepNettsException, IOException {
        
    //    DeepNetts.getInstance().setMaxThreads(1);
        
        int numInputs= 29;
        int numOutputs = 1;
        boolean hasHeader = true;

        // load spam data  set from csv file 
        DataSet dataSet = DataSets.readCsv("datasets/creditcard-balanced.csv", numInputs, numOutputs, hasHeader);
        
        // scale data to [0, 1] range which is used by neural network
        DataSets.scaleToMax(dataSet);
        
        // split data into training and test set
        DataSet<MLDataItem>[] trainTestSet = dataSet.split(0.6);
        DataSet<MLDataItem> trainingSet = trainTestSet[0];
        DataSet<MLDataItem> testSet = trainTestSet[1];
        
        // create instance of feed forward neural network using its builder
        FeedForwardNetwork neuralNet = FeedForwardNetwork.builder()
                .addInputLayer(numInputs)   // size of the input layer corresponds to number of inputs
                .addFullyConnectedLayer(80, ActivationType.TANH) 
                .addOutputLayer(numOutputs, ActivationType.SIGMOID) // size of output layer corresponds to number of outputs, which is 1 for binary classification problems, and sigmoid transfer function is used for binary classification
                .lossFunction(LossType.CROSS_ENTROPY) // cross entropy loss function is commonly used for classification problems
                .build();
  
        // set parameters of the training algorithm
        neuralNet.getTrainer().setStopError(0.02f)
                              .setStopEpochs(10000)
                              .setLearningRate(0.001f);
     
        neuralNet.train(trainingSet);
        
        // test neural network and print evaluation metrics
        EvaluationMetrics em = neuralNet.test(testSet);
        System.out.println(em);
                
        // Example usage of the trained network with vis rec api
        BinaryClassifier<float[]> binClassifier = new FeedForwardNetBinaryClassifier(neuralNet);    
        float[] testTransaction = testSet.get(0).getInput().getValues();
        

        Float result = binClassifier.classify(testTransaction);
        System.out.println("Fraud probability: "+result);            
            
    }
}