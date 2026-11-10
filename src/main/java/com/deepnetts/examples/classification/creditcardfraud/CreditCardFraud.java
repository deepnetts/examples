package com.deepnetts.examples.classification.creditcardfraud;

import com.deepnetts.examples.classification.creditcardfraud.dto.FraudPrediction;
import com.deepnetts.examples.classification.creditcardfraud.dto.Transaction;
import com.deepnetts.examples.classification.creditcardfraud.pipeline.FraudPredictor;
import com.deepnetts.examples.classification.creditcardfraud.pipeline.TransactionEncoder;
import com.deepnetts.examples.classification.creditcardfraud.pipeline.TransactionValidator;

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

// Fraud detection example: a neural network learns from past transactions.
// Then it tells you how likely a new transaction is to be fraud.
public class CreditCardFraud {

    private static final String DATASET = "datasets/creditcard-fraud.csv";
    private static final String MODEL   = "models/credit-card-fraud.dnet";
    private static final int    SEED    = 42;

    // At or above this probability (0 to 1), a transaction counts as fraud.
    // Lower = catches more fraud, but also raises more false alarms.
    // Higher = fewer false alarms, but more fraud slips through.
    // 0.5 is the neutral middle; move it when one kind of mistake costs more than the other.
    // Used in both the evaluation and the prediction.
    private static final float THRESHOLD = 0.5f;

    // Hides the detailed training log.
    private static final Logger DEEPNETTS_LOG = Logger.getLogger("deepnetts");

    public static void main(String[] args) {
        // Prints a clear error (missing file, bad CSV row, invalid YourTransaction) and stops.
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

        // Validates YourTransaction before training starts.
        Transaction transaction = YourTransaction.create();
        TransactionValidator.validate(transaction, "YourTransaction");

        // Turns CSV rows and transactions into the numbers the network learns from (see pipeline/TransactionEncoder).
        TransactionEncoder encoder = new TransactionEncoder();

        // ── 1. Data ──────────────────────────────
        // Loads the CSV: each row becomes a transaction (the input) and its label, the correct answer
        // (is_fraud: 1 = fraud, 0 = legitimate).
        TabularDataSet<MLDataItem> data = encoder.load(DATASET);

        // Counts the fraud rows (label 1), for the dataset summary.
        int fraudCount = 0;
        for (MLDataItem item : data) {
            fraudCount += item.getTargetOutput().getValues()[0] == 1f ? 1 : 0;
        }
        Report.dataset(data.size(), fraudCount);

        // Splits the data: 80% to train on, 20% to test on transactions the network never saw.
        // 80/20 is a common choice: enough rows to learn from, and enough left for an honest test.
        TrainTestSplit split = data.trainTestSplit(0.8);
        DataSet<MLDataItem> trainingSet = split.getTrainingSet();
        DataSet<MLDataItem> testSet     = split.getTestSet();

        // Puts all inputs on the same 0-1 scale, so no input dominates.
        // Important: set the scale on the training set only. If the test set helped set it, the test would already
        // know something about its own rows (data leakage), and the results would look better than they are.
        MinMaxScaler scaler = DataSets.scaleToMinMax(trainingSet);
        scaler.apply(testSet);

        // ── 2. Training ──────────────────────────
        Report.trainingStarted();

        // Builds the network layer by layer using the builder pattern.
        FeedForwardNetwork network = FeedForwardNetwork.builder()
                // Input layer: receives the transaction numbers.
                .addInputLayer(TransactionEncoder.NUM_INPUTS)
                // Hidden layer: 10 units that look for patterns in those numbers.
                // A small number is enough for a small problem; try 20 and compare the results.
                // Tanh squeezes each unit's output into the range -1 to 1.
                .addFullyConnectedLayer(10, ActivationType.TANH)
                // Output layer: 1 number, the fraud probability: a score from 0 to 1.
                // The network learned from data with 42% fraud, so the score is higher than the real chance of fraud
                // (real card data has well under 1%). Read it as "how suspicious", not as a real probability.
                // Sigmoid squeezes the output into 0 to 1.
                .addOutputLayer(TransactionEncoder.NUM_OUTPUTS, ActivationType.SIGMOID)
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
                // Adam: a popular way to adjust the network after each step; it adapts the step size by itself.
                .setOptimizer(OptimizerType.ADAM)
                // Shuffles the examples each epoch, so the network doesn't learn their order.
                .setShuffle(true)
                // Early stopping: ends training when the results on the held-out data stop improving.
                // Deep Netts then keeps the network from the last epoch, not the best one.
                .setEarlyStopping(true)
                // Checks the results every 5 epochs.
                .setCheckpointEpochs(5)
                // Stops after 2 checks in a row without improvement (so at least 10 epochs after the best one).
                .setEarlyStoppingPatience(2);

        // Trains the network.
        // 20% of the training data is set aside (held-out data): it is used only to check progress, not to learn.
        network.getTrainer().train(trainingSet, 0.2);
        Report.trainingFinished(network.getTrainer().getCurrentEpoch());

        // ── 3. Evaluation ────────────────────────
        // Checks how well the network works on the 20% of transactions it never saw.
        ClassifierEvaluator evaluator = new ClassifierEvaluator();
        evaluator.setThreshold(THRESHOLD);
        ClassificationMetrics metrics = evaluator.evaluate(network, testSet);
        ConfusionMatrix confusionMatrix = evaluator.getConfusionMatrix();
        Report.evaluation(confusionMatrix, metrics, THRESHOLD);

        // ── 4. Save, load and predict ────────────
        // Saves the network together with its scale to models/credit-card-fraud.dnet.
        // It can be used later without training again.
        network.setNormalizer(scaler);
        Files.createDirectories(Path.of(MODEL).getParent());
        network.save(MODEL);

        // Loads the saved network and predicts YourTransaction, without training again.
        FeedForwardNetwork loadedNetwork = NeuralNetwork.load(MODEL, FeedForwardNetwork.class);
        FraudPrediction prediction = new FraudPredictor(loadedNetwork, encoder, THRESHOLD).predict(transaction);
        Report.prediction(transaction, prediction);
    }
}
