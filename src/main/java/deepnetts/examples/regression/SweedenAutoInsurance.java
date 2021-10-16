package deepnetts.examples.regression;

import deepnetts.data.DataSets;
import javax.visrec.ml.eval.EvaluationMetrics;
import deepnetts.eval.Evaluators;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.net.train.BackpropagationTrainer;
import java.io.IOException;
import java.util.Arrays;
import javax.visrec.ml.data.DataSet;


/**
 * Predicting the total payment for all claims (in thousands of Swedish Kronor), given the total number of claims.
 * This is a minimal example for simple linear regression using FeedForwardNetwork.
 * Linear regression finds an optimal position of straight line through the given example data, so the total difference between line and data is minimal.
 * Uses a single layer with one output and linear activation function, and Mean Squared Error for Loss function.
 * You can use linear regression to roughly estimate a global trend in data.
 *
 * TODO: dont print accuracy for regression problems!
 *
 * For the best performance and accuracy the recommended way to run this example is to use Deep Netts Pro, with Free Development License.
 * https://www.deepnetts.com/download
 *
 * Step-by-step guide for setting up Deep Netts is available at
 * https://www.deepnetts.com/getting-started
 * 
 * @see FeedForwardNetwork
 *
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class SweedenAutoInsurance {

    public static void main(String[] args) throws IOException {

        String datasetFile = "datasets/SweedenAutoInsurance.csv";
        int inputsNum = 1;
        int outputsNum = 1;

        DataSet dataSet = DataSets.readCsv(datasetFile, inputsNum, outputsNum);
        DataSet[] trainTestPairSet = dataSet.split(0.7, 0.3);

        FeedForwardNetwork neuralNet = FeedForwardNetwork.builder()
                .addInputLayer(1)
                .addOutputLayer(1, ActivationType.LINEAR)
                .lossFunction(LossType.MEAN_SQUARED_ERROR)
                .build();

        BackpropagationTrainer trainer = neuralNet.getTrainer();
        trainer.setMaxError(0.0001f)
               .setMaxEpochs(100)
               .setLearningRate(0.1f);

        neuralNet.train(trainTestPairSet[0]);
               
       // evaluate model with test set 
       EvaluationMetrics em = neuralNet.test(trainTestPairSet[1]);
       System.out.println(em);

       // use model for prediction
       float[] predictedOutput = neuralNet.predict(0.153225806f);
       System.out.println(Arrays.toString(predictedOutput));
    }

}
