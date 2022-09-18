package deepnetts.examples.classification;

import deepnetts.data.DataSets;
import deepnetts.data.TrainTestPair;
import javax.visrec.ml.eval.EvaluationMetrics;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.net.train.BackpropagationTrainer;
import java.io.IOException;
import java.util.Arrays;
import javax.visrec.ml.data.DataSet;

/**
 * Recognition of sonar signals.
 * This example shows how to perform binary classification based on logistic regression using FeedForwardNetwork.
 * The trained model predict weather the given input sonar signal represents a mine, or not.
 *
 * Data Set Description
 * The data set contains 111 patterns obtained by bouncing sonar signals off a metal cylinder at various angles and under various conditions, 
 * and 97 patterns obtained from rocks under similar conditions.
 * Each pattern is a set of 60 numbers in the range 0.0 to 1.0. Each number represents the energy within a particular frequency band.
 * Original data set: http://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks)
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
public class LogisticRegression {

    public static void main(String[] args) throws IOException {

        int numInputs = 60;
        int numOutputs = 1;
        
        DataSet dataSet = DataSets.readCsv("datasets/sonar.csv", numInputs, numOutputs);
        TrainTestPair trainTestPair = DataSets.trainTestSplit(dataSet, 0.7);

        FeedForwardNetwork neuralNet = FeedForwardNetwork.builder()
                .addInputLayer(numInputs)                               // width of the input layer should be same as number of inputs
                .addOutputLayer(numOutputs, ActivationType.SIGMOID)     // width of the output layer should be same as the number of outputs, and with SIGMOID (logistic) function since we're doing logistic regression
                .lossFunction(LossType.CROSS_ENTROPY)                   // Cross Entropy Loss function is used for classification problems
                .build();

        BackpropagationTrainer trainer = neuralNet.getTrainer();
        trainer.setLearningRate(0.01f)
               .setStopError(0.1f)
               .setStopEpochs(20000);

        neuralNet.train(trainTestPair.getTrainingeSet());

        EvaluationMetrics testResults = neuralNet.test(trainTestPair.getTestSet());
        System.out.println(testResults);

        float[] predictedOut = neuralNet.predict(new float[]{0.0303f,0.0353f,0.0490f,0.0608f,0.0167f,0.1354f,0.1465f,0.1123f,0.1945f,0.2354f,0.2898f,0.2812f,0.1578f,0.0273f,0.0673f,0.1444f,0.2070f,0.2645f,0.2828f,0.4293f,0.5685f,0.6990f,0.7246f,0.7622f,0.9242f,1.0000f,0.9979f,0.8297f,0.7032f,0.7141f,0.6893f,0.4961f,0.2584f,0.0969f,0.0776f,0.0364f,0.1572f,0.1823f,0.1349f,0.0849f,0.0492f,0.1367f,0.1552f,0.1548f,0.1319f,0.0985f,0.1258f,0.0954f,0.0489f,0.0241f,0.0042f,0.0086f,0.0046f,0.0126f,0.0036f,0.0035f,0.0034f,0.0079f,0.0036f,0.0048f});
        System.out.println(Arrays.toString(predictedOut));

    }

}