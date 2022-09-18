package deepnetts.examples.classification;

import deepnetts.core.DeepNetts;
import deepnetts.data.DataSets;
import deepnetts.eval.Evaluators;
import javax.visrec.ml.eval.EvaluationMetrics;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.util.DeepNettsException;
import java.io.IOException;
import javax.visrec.ml.data.DataSet;
import deepnetts.data.MLDataItem;
import deepnetts.data.norm.MaxScaler;
import javax.visrec.ml.classification.BinaryClassifier;
import javax.visrec.ri.ml.classification.FeedForwardNetBinaryClassifier;

/**
 * Spam  Classification example.
 * This example shows how to create binary classifier for spam classification, using Feed Forward neural network.
 * Data is given as CSV file.
 *
  * For the best performance and accuracy the recommended way to run this example is to use Deep Netts Pro, with Free Development License.
 * https://www.deepnetts.com/download
 *
 * Step-by-step guide for setting up Deep Netts is available at
 * https://www.deepnetts.com/getting-started
 * 
 * @see FeedForwardNetwork
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class SpamClassifier {

    public static void main(String[] args) throws DeepNettsException, IOException {

        int numInputs = 57;
        int numOutputs = 1;
        
        // load spam data  set from csv file
        DataSet dataSet = DataSets.readCsv("datasets/spam.csv", numInputs, numOutputs, true);             

        // split data set into train and test set
        DataSet<MLDataItem>[] trainTest = dataSet.split(0.6, 0.4);
        
        // normalize/scale training and test data
        MaxScaler scaler = new MaxScaler(trainTest[0]);
        scaler.apply(trainTest[0]);
        scaler.apply(trainTest[1]);
        
        // create instance of feed forward neural network using its builder
        FeedForwardNetwork neuralNet = FeedForwardNetwork.builder()
                .addInputLayer(numInputs)
                .addFullyConnectedLayer(25, ActivationType.TANH)
                .addOutputLayer(numOutputs, ActivationType.SIGMOID)
                .lossFunction(LossType.CROSS_ENTROPY)
                .randomSeed(123)
                .build();

        // set training settings
        neuralNet.getTrainer().setStopError(0.3f)
                              .setLearningRate(0.001f);
        
        // start training
        neuralNet.train(trainTest[0]);
        
        // test network /  evaluate classifier
        EvaluationMetrics em = Evaluators.evaluateClassifier(neuralNet, trainTest[1]);
        System.out.println(em);
        
        // create binary classifier using trained network
        BinaryClassifier<float[]> binClassifier = new FeedForwardNetBinaryClassifier(neuralNet);
        
        // get single feature array from test set
        float[] testEmail = trainTest[1].get(0).getInput().getValues();
        // feed the classifer and get result / spam probability
        Float result = binClassifier.classify(testEmail);
        System.out.println("Spam probability: "+result);      
        
        // shutdown the thread pool
        DeepNetts.shutdown();        
    }
    

}