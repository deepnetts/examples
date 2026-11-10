package com.deepnetts.examples.classification.sonarmines;

import com.deepnetts.examples.classification.sonarmines.dto.MinePrediction;
import com.deepnetts.examples.classification.sonarmines.dto.SonarContact;
import com.deepnetts.examples.classification.sonarmines.pipeline.ContactEncoder;
import com.deepnetts.examples.classification.sonarmines.pipeline.ContactValidator;
import com.deepnetts.examples.classification.sonarmines.pipeline.MinePredictor;

import deepnetts.core.DeepNetts;
import deepnetts.data.DataSets;
import deepnetts.data.MLDataItem;
import deepnetts.data.TabularDataSet;
import deepnetts.data.TrainTestSplit;
import deepnetts.data.norm.MinMaxScaler;
import deepnetts.eval.ClassificationMetrics;
import deepnetts.eval.ClassifierEvaluator;
import deepnetts.eval.ConfusionMatrix;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.NeuralNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.net.train.opt.OptimizerType;
import deepnetts.util.RandomGenerator;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.visrec.ml.data.DataSet;

// Sonar mine detection example: a neural network learns from past sonar contacts.
// Then it tells you how likely a new contact is to be a mine.
public class SonarMines {

    private static final String DATASET = "datasets/sonar-mines.csv";
    private static final String MODEL   = "models/sonar-mines.dnet";
    private static final int    SEED    = 42;

    // At or above this probability (0 to 1), a contact counts as a mine.
    // Lower = catches more mines, but also raises more false alarms.
    // Higher = fewer false alarms, but more mines slip through.
    // Used in both the evaluation and the prediction.
    private static final float THRESHOLD = 0.5f;

    // Hides the detailed training log.
    private static final Logger DEEPNETTS_LOG = Logger.getLogger("deepnetts");

    public static void main(String[] args) {
        // Prints a clear error (missing file, bad CSV row, invalid YourContact) and stops.
        try {
            run();
        } catch (IOException | IllegalArgumentException e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
        } catch (ClassNotFoundException e) {
            // The saved model names a class that is not on the classpath (for example another Deep Netts version).
            System.err.println("Error: cannot load the saved model, class not found: " + e.getMessage());
            System.exit(1);
        }
    }

    static void run() throws IOException, ClassNotFoundException {

        // Fixed seed and a single thread, so you get the same results on every run.
        RandomGenerator.getDefault().initSeed(SEED);
        DeepNetts.getInstance().setMaxThreads(1);
        // Comment out this line to see the detailed training log.
        DEEPNETTS_LOG.setLevel(Level.WARNING);

        // Validates YourContact before training starts.
        SonarContact contact = YourContact.create();
        ContactValidator.validate(contact, "YourContact");

        ContactEncoder encoder = new ContactEncoder();

        // ── 1. Data ──────────────────────────────
        // Loads the CSV: each row becomes a sonar contact + a label (is_mine).
        TabularDataSet<MLDataItem> data = encoder.load(DATASET);

        // Counts the rows with label 1, for the dataset summary.
        int mineCount = 0;
        for (MLDataItem item : data) {
            mineCount += item.getTargetOutput().getValues()[0] == 1f ? 1 : 0;
        }
        Report.dataset(data.size(), mineCount);

        // Splits the data: 80% for training, 20% for the final test on contacts the network never saw.
        TrainTestSplit split = data.trainTestSplit(0.8);
        DataSet<MLDataItem> trainingSet = split.getTrainingSet();
        DataSet<MLDataItem> testSet     = split.getTestSet();

        // Puts all inputs on the same 0-1 scale, so no input dominates.
        // Important: calculate the min and max from the training set only, not the test set (data leakage).
        MinMaxScaler scaler = DataSets.scaleToMinMax(trainingSet);
        scaler.apply(testSet);

        // ── 2. Training ──────────────────────────
        Report.trainingStarted();

        // Builds the network layer by layer using the builder pattern.
        FeedForwardNetwork network = FeedForwardNetwork.builder()
                // Input layer: receives the sonar contact numbers.
                .addInputLayer(ContactEncoder.NUM_INPUTS)
                // Hidden layer: 10 units that look for patterns in those numbers.
                // TANH is an activation function: it squeezes each unit's output into the range -1 to 1.
                .addFullyConnectedLayer(10, ActivationType.TANH)
                // Output layer: 1 number (the mine probability).
                // SIGMOID squeezes the output into 0 to 1, so it can be read as a probability.
                .addOutputLayer(ContactEncoder.NUM_OUTPUTS, ActivationType.SIGMOID)
                // Cross entropy measures how wrong the predictions are.
                .lossFunction(LossType.CROSS_ENTROPY)
                .build();

        network.getTrainer()
                // An epoch is one pass over the data.
                .setStopEpochs(1000)
                // The learning rate is the step size.
                // Too small = learning is very slow.
                // Too large = the network jumps past the best solution.
                .setLearningRate(0.01f)
                // Adam: an algorithm that improves the network step by step, adapting the step size as it goes.
                .setOptimizer(OptimizerType.ADAM)
                // Shuffles the examples each epoch, so the network doesn't learn their order.
                .setShuffle(true)
                // Early stopping: ends training when the results on the held-out data stop improving.
                // Deep Netts then keeps the network from the last epoch, not the best one.
                .setEarlyStopping(true)
                // Checks the results every 5 epochs.
                .setCheckpointEpochs(5)
                // Stops after 2 checks in a row without improvement.
                .setEarlyStoppingPatience(2);

        // Trains the network.
        // 20% of the training data is set aside (held-out data): it is used only to check progress, not to learn.
        // The test set stays untouched until the final evaluation.
        network.getTrainer().train(trainingSet, 0.2);
        Report.trainingFinished(network.getTrainer().getCurrentEpoch());

        // ── 3. Evaluation ────────────────────────
        // Checks how well the network works on the 20% of contacts it never saw.
        ClassifierEvaluator evaluator = new ClassifierEvaluator();
        evaluator.setThreshold(THRESHOLD);
        ClassificationMetrics metrics = evaluator.evaluate(network, testSet);
        ConfusionMatrix cm = evaluator.getConfusionMatrix();
        Report.evaluation(cm, metrics, THRESHOLD);

        // ── 4. Save, load and predict ────────────
        // Saves the network together with its scale to models/sonar-mines.dnet.
        // It can be used later without training again.
        network.setNormalizer(scaler);
        Files.createDirectories(Path.of(MODEL).getParent());
        network.save(MODEL);

        // Loads the saved network and predicts YourContact, without training again.
        FeedForwardNetwork loadedNetwork = NeuralNetwork.load(MODEL, FeedForwardNetwork.class);
        MinePrediction prediction = new MinePredictor(loadedNetwork, encoder, THRESHOLD).predict(contact);
        Report.prediction(contact, prediction);
    }
}
