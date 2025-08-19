package deepnetts.examples.misc;

import deepnetts.core.DeepNetts;
import deepnetts.data.DataSets;
import deepnetts.data.TabularDataSet;
import deepnetts.data.norm.MaxScaler;
import deepnetts.eval.ClassifierEvaluator;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.net.train.BackpropagationTrainer;
import deepnetts.net.train.KFoldCrossValidation;
import java.io.IOException;
import javax.visrec.ml.data.DataSet;
import javax.visrec.ml.eval.EvaluationMetrics;

/**
 * This example shows how to perform K-fold cross-validation.
 * In k-fold cross-validation training is repeated k times with k different subsets of data for evaluation.
 *
 * For the best performance and accuracy the recommended way to run this example is to use Deep Netts Pro, with Free Development License.
 * https://www.deepnetts.com/download
 *
 * Step-by-step guide for setting up Deep Netts is available at
 * https://www.deepnetts.com/quickstart
 * 
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class KFoldCrossvalidationDemo {

    public static void main(String[] args) throws IOException {
        TabularDataSet dataSet = DataSets.readCsv("datasets/iris-flowers.csv", 4, 3, true);
        dataSet.setColumnNames(new String[] {"petal width","petal height","sepal width","sepal height", "Setose", "Vrsicolor", "Virginica"});

        MaxScaler norm = new MaxScaler(dataSet);
        norm.apply(dataSet);   

        // create model to train
        FeedForwardNetwork neuralNet = FeedForwardNetwork.builder()
                                            .addInputLayer(4)
                                            .addFullyConnectedLayer(20, ActivationType.TANH)
                                            .addOutputLayer(3, ActivationType.SOFTMAX)
                                            .lossFunction(LossType.CROSS_ENTROPY)
                                            .randomSeed(123)
                                            .build();

        BackpropagationTrainer trainer = neuralNet.getTrainer();
        trainer.setStopError(0.01f)
               .setLearningRate(0.1f)
               .setStopEpochs(2000);

        ClassifierEvaluator evaluator = new ClassifierEvaluator();

        KFoldCrossValidation kfcv = KFoldCrossValidation.builder()
                                        .model(neuralNet)
                                        .trainingSet(dataSet)
                                        .numSplits(5)
                                        .evaluator(evaluator)
                                        .build();

        kfcv.run();
        EvaluationMetrics em = kfcv.getMacroAverage();
        System.out.println(em);
        
        FeedForwardNetwork bestNetwork = (FeedForwardNetwork)kfcv.getBestNetwork();
        EvaluationMetrics bestEm = kfcv.getBestResult();
        
        System.out.println("Best evaluation metrics:");
        System.out.println(bestEm);
               
        
    }

}