package deepnetts.examples.regression;

import deepnetts.core.DeepNetts;
import deepnetts.data.DataSets;
import deepnetts.data.MLDataItem;
import deepnetts.data.norm.MaxNormalizer;
import javax.visrec.ml.eval.EvaluationMetrics;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.net.train.BackpropagationTrainer;
import deepnetts.util.FileIO;
import deepnetts.util.Tensor;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.visrec.ml.data.DataSet;

/**
 * Predicting CPU performance using deep learning in Java.
 * This example shows how to create regression model using FeedForwardNetwork

 * Original Data Set: https://archive.ics.uci.edu/ml/datasets/Computer+Hardware
 * 
 * Data set description
 *
 * 1. MYCT: machine cycle time in nanoseconds (integer)
 * 2. MMIN: minimum main memory in kilobytes (integer)
 * 3. MMAX: maximum main memory in kilobytes (integer)
 * 4. CACH: cache memory in kilobytes (integer)
 * 5. CHMIN: minimum channels in units (integer)
 * 6. CHMAX: maximum channels in units (integer)
 * 7. PRP: published relative performance (integer)  - value to predict!
 * 
 * All attributes are given in CSV file
 * 
 * For the best performance and accuracy the recommended way to run this example is to use Deep Netts Pro, with Free Development License.
 * https://www.deepnetts.com/download
 *
 * Step-by-step guide for setting up Deep Netts is available at
 * https://www.deepnetts.com/getting-started
 * 
 * @see FeedForwardNetwork
 * 
 */
public class CpuPerformancePrediction {

    public static void main(String[] args) throws IOException {

            int inputsNum = 6;  // the model should use 6 input parameters (first 6 values in a row of csv file)
            int outputsNum = 1; // to predict a single value:  published relative performance of the CPU (7th value in the row of csv file)
            String csvFilename = "datasets/cpu_data.csv"; // the csv file with data to build the model

            // load and create data set from csv file
            DataSet<MLDataItem> dataSet = DataSets.readCsv(csvFilename , inputsNum, outputsNum, true);
            
            // normalize data - scale to range [0, 1] in order to feed the neural network
            MaxNormalizer norm = new MaxNormalizer(dataSet);
            norm.normalize(dataSet);
    
            // split data set into the training and test set with 70% for training and 30% for test
            DataSet<MLDataItem>[] trainTest = dataSet.split(0.7, 0.3);  
            DataSet<MLDataItem> trainingSet = trainTest[0];
            DataSet<MLDataItem> testSet = trainTest[1];
            
            // create neural network using builder
            FeedForwardNetwork neuralNet = FeedForwardNetwork.builder()
                    .addInputLayer(inputsNum)
                    .addHiddenFullyConnectedLayers(12, 12)
                    .addOutputLayer(outputsNum, ActivationType.LINEAR)
                    .hiddenActivationFunction(ActivationType.TANH)
                    .lossFunction(LossType.MEAN_SQUARED_ERROR)
                    .build();

            // set parameters of the training algorithm
            BackpropagationTrainer trainer = neuralNet.getTrainer();
            trainer.setMaxError(0.001f) // error threshold when algorithm stops
                   .setMaxEpochs(1000)  // stop th etraining if the error does not fall under maxError after maxEpochs training iterations
                   .setLearningRate(0.01f); // size of the learning step in each iteration

            // train network using training set
            neuralNet.train(trainingSet);

            // test network using test set and print testing/evaluation results
            EvaluationMetrics em = neuralNet.test(testSet);
            System.out.println(em);

            // perform prediction for the given input value
            neuralNet.setInput(testSet.get(0).getInput());            
            float[] predictedOutput = neuralNet.getOutput();
            
            // create a tensor object from given output array
            Tensor predictedTensor = Tensor.of(predictedOutput);
            norm.deNormalizeOutputs(predictedTensor);
            
            // de-normalize output here
            System.out.println("Predicted output:" + predictedTensor.toString());
            
            // save network to file
            FileIO.writeToFile(neuralNet, "savedNetwork.dnet");
            
            // load saved trained network
            try {
                FeedForwardNetwork neuralNet2 =  FileIO.createFromFile("savedNetwork.dnet", FeedForwardNetwork.class);
                // now you can use neuralNet2 for prediction as shown above
            } catch (ClassNotFoundException ex) { 
                Logger.getLogger(CpuPerformancePrediction.class.getName()).log(Level.SEVERE, null, ex);
            } 

        // shutdown the thread pool
        DeepNetts.shutdown();            
    }

}
