package deepnetts.examples.classification;

import deepnetts.core.DeepNetts;
import deepnetts.data.DataSets;
import deepnetts.data.TrainTestSplit;
import deepnetts.data.norm.MaxScaler;
import deepnetts.eval.ClassificationMetrics;
import deepnetts.eval.Evaluators;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.net.train.BackpropagationTrainer;
import deepnetts.util.DeepNettsException;
import deepnetts.util.FileIO;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.visrec.ml.data.DataSet;

/**
 * Iris Flowers Classification. 
 * This example shows hot to train Deed Forward Neural Network to perform  multi class classification.
 * Multi class classification is assigning input to one of several possible categories/classes.
 * In this example Inputs are 4 dimensions of iris flower, and the output is one of 3 possible categories/classes.
 * 
 * Data Set Description:
 * 150 examples (flowers)
 * 4 input attributes: sepal_length, sepal_width, petal_length, petal_width
 * 1 target attribute: flower category/class with three possible values setosa, versicolor, virginica
 * When working with neural networks multiple categories are converted one-hot-encoding, where each category corresponds to
 * binary vector, with all zeros and 1 only on one position that corresponds to specific category. So caalled one-hot-encoding.
 *
 * URL: https://en.wikipedia.org/wiki/Iris_flower_data_set
 * 
 * For the best performance and accuracy the recommended way to run this example is to use Deep Netts Pro, with Free Development License.
 * https://www.deepnetts.com/download
 *
 * Step-by-step guide for setting up Deep Netts is available at
 * https://www.deepnetts.com/quickstart
 * 
 * @see FeedForwardNetwork
 */
public class IrisFlowersClassificationAdvanced {

    static final Logger LOGGER = Logger.getLogger(DeepNetts.class.getName());

    public static void main(String[] args) throws DeepNettsException, IOException {
        // load iris data  set
        DataSet dataSet = DataSets.readCsv("datasets/iris-flowers.csv", 4, 3, true, ",");
        TrainTestSplit trainTest = DataSets.trainTestSplit(dataSet, 0.65);

        // normalize data using max normalization
        MaxScaler scaler = new MaxScaler(trainTest.getTrainingSet());
        scaler.apply(trainTest.getTrainingSet());   
        scaler.apply(trainTest.getTestSet());
        
        // create an instance of a neural network  using builder
        FeedForwardNetwork neuralNet = FeedForwardNetwork.builder()
                .addInputLayer(4)
                .addFullyConnectedLayer(16, ActivationType.TANH)
                .addOutputLayer(3, ActivationType.SOFTMAX)
                .lossFunction(LossType.CROSS_ENTROPY)
                .randomSeed(123).
                build();
        
        // get and configure an instanceof training algorithm
        BackpropagationTrainer trainer = neuralNet.getTrainer();
        trainer.setStopError(0.05f)
               .setStopEpochs(1000000)
               .setLearningRate(0.1f)
               .setEarlyStopping(true)  // early stopping will stop the training if the error starts decreasing to slow
               .setEarlyStoppingMinLossChange(0.001f)   // stop the training if training error is smaller then this value
             //  .setLearningRateDecay(0.01f)       // uncoment these to use these additional training settings
               //.setOptimizer(OptimizerType.MOMENTUM)
             //  .setMomentum(0.9f);                
               .setBatchMode(false);

                      
       // trainer.setExtendedLogging(true);
             
        trainer.train(trainTest.getTrainingSet());

        ClassificationMetrics ceResult = Evaluators.evaluateClassifier(neuralNet, trainTest.getTestSet());
        
        LOGGER.info("Classification performance measure" + System.lineSeparator());
        LOGGER.info(ceResult.toString());
        
       
        FileIO.writeToFile(neuralNet, "testSerialization.nnet");
        try {
            FeedForwardNetwork loadedNet = FileIO.createFromFile("testSerialization.nnet", FeedForwardNetwork.class);
            ceResult = Evaluators.evaluateClassifier(loadedNet, trainTest.getTestSet());
            LOGGER.info("Classification performance measure" + System.lineSeparator());
            LOGGER.info(ceResult.toString());            
        } catch (ClassNotFoundException ex) {
            Logger.getLogger(IrisFlowersClassificationAdvanced.class.getName()).log(Level.SEVERE, null, ex);
        }
                 
    }

}
