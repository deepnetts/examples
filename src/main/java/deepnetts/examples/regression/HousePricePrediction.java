package deepnetts.examples.regression;

import deepnetts.core.DeepNetts;
import deepnetts.data.DataSets;
import deepnetts.data.TabularDataSet;
import deepnetts.eval.Evaluators;
import javax.visrec.ml.eval.EvaluationMetrics;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.util.Tensor;
import java.io.IOException;
import javax.visrec.ml.data.DataSet;

/**
 * Minimal example for regression using FeedForwardNetwork with one hidden layer.
 *
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class HousePricePrediction {

    public static void main(String[] args) throws IOException {

            int inputsNum = 1;
            int outputsNum = 1;
            String csvFilename = "datasets/bostonhousing.csv";

            // load and create data set from csv file
            TabularDataSet dataSet = DataSets.readCsv(csvFilename , inputsNum, outputsNum, true);
            DataSet[] trainTestPairSet = dataSet.split(0.6);

            // create neural network using network specific builder
            FeedForwardNetwork neuralNet = FeedForwardNetwork.builder()
                    .addInputLayer(inputsNum)
                    .addHiddenFullyConnectedLayers(40, 20)
                    .addOutputLayer(outputsNum, ActivationType.LINEAR)
                    .hiddenActivationFunction(ActivationType.TANH)                    
                    .lossFunction(LossType.MEAN_SQUARED_ERROR)                    
                    .build();
            
            neuralNet.getTrainer().setMaxError(0.006f)
                                  .setLearningRate(0.0001f);
            neuralNet.train(trainTestPairSet[0]);

            EvaluationMetrics pm = Evaluators.evaluateRegressor(neuralNet, trainTestPairSet[1]);
            System.out.println(pm);

            // perform prediction for some input value
            neuralNet.setInput(Tensor.create(1, 1, new float[] {0.2f}));
            System.out.println("Predicted price of the house is for 8 :" + neuralNet.getOutput()[0]);//*50);
            
            // shutdown the thread pool
            DeepNetts.shutdown();            
    }

}
