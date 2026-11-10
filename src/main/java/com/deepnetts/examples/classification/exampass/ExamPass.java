package com.deepnetts.examples.classification.exampass;

import com.deepnetts.examples.classification.exampass.dto.PassPrediction;
import com.deepnetts.examples.classification.exampass.dto.PassWeights;
import com.deepnetts.examples.classification.exampass.dto.Student;
import com.deepnetts.examples.classification.exampass.pipeline.PassPredictor;
import com.deepnetts.examples.classification.exampass.pipeline.StudentEncoder;
import com.deepnetts.examples.classification.exampass.pipeline.StudentValidator;

import deepnetts.core.DeepNetts;
import deepnetts.data.MLDataItem;
import deepnetts.data.TabularDataSet;
import deepnetts.data.TrainTestSplit;
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

// Logistic regression example: a network learns who passes an exam, from study hours and practice tests.
// The network has no hidden layer and a SIGMOID output: that is logistic regression, the simplest yes/no model.
// At the end, the example reads out of the network how much each input pushes toward passing.
public class ExamPass {

    private static final String DATASET = "datasets/exam-pass.csv";
    private static final String MODEL   = "models/exam-pass.dnet";
    private static final int    SEED    = 42;

    // At or above this probability (0 to 1), a student counts as passing.
    // 0.5 is the neutral middle: neither mistake (a pass called a fail, or the other way round) costs more here.
    private static final float THRESHOLD = 0.5f;

    // The rule the data was made with (see dataset-tools/make_exam_pass.py), before chance decides.
    // Real data doesn't tell you its rule: here it shows how close the network gets.
    private static final PassWeights RULE_OF_THE_DATA = new PassWeights(-6.0, 0.15, 0.60);

    // Hides the detailed training log.
    private static final Logger DEEPNETTS_LOG = Logger.getLogger("deepnetts");

    public static void main(String[] args) {
        // Prints a clear error (missing file, bad CSV row, invalid YourStudent) and stops.
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

        // Validates YourStudent before training starts.
        Student student = YourStudent.create();
        StudentValidator.validate(student, "YourStudent");

        // Turns CSV rows and students into the numbers the network learns from (see pipeline/StudentEncoder).
        StudentEncoder encoder = new StudentEncoder();

        // ── 1. Data ──────────────────────────────
        // Loads the CSV: each row becomes a student (the input) and its label, the correct answer
        // (passed: 1 = passed, 0 = failed).
        TabularDataSet<MLDataItem> data = encoder.load(DATASET);

        // Counts the students who passed (label 1), for the dataset summary.
        int passedCount = 0;
        for (MLDataItem item : data) {
            passedCount += item.getTargetOutput().getValues()[0] == 1f ? 1 : 0;
        }
        Report.dataset(data.size(), passedCount);

        // Splits the data: 80% to train on, 20% to test on students the network never saw.
        TrainTestSplit split = data.trainTestSplit(0.8);
        DataSet<MLDataItem> trainingSet = split.getTrainingSet();
        DataSet<MLDataItem> testSet     = split.getTestSet();

        // ── 2. Training ──────────────────────────
        Report.trainingStarted();

        // Builds the network layer by layer using the builder pattern.
        FeedForwardNetwork network = FeedForwardNetwork.builder()
                // Input layer: receives the two student numbers (study hours and practice tests).
                .addInputLayer(StudentEncoder.NUM_INPUTS)
                // Output layer: 1 number, the probability to pass. There is no hidden layer:
                // the network adds up bias + weight1 * hours + weight2 * tests, and SIGMOID squeezes the sum into 0 to 1.
                // That is exactly logistic regression.
                .addOutputLayer(StudentEncoder.NUM_OUTPUTS, ActivationType.SIGMOID)
                // Cross entropy measures how wrong the predicted probabilities are.
                .lossFunction(LossType.CROSS_ENTROPY)
                .build();

        network.getTrainer()
                // At most 2000 epochs (passes over the data); early stopping usually ends much sooner.
                .setStopEpochs(2000)
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
                // Stops after 4 checks in a row without improvement (so at least 20 epochs after the best one).
                .setEarlyStoppingPatience(4);

        // Trains the network.
        // 20% of the training data is set aside (held-out data): it is used only to check progress, not to learn.
        network.getTrainer().train(trainingSet, 0.2);
        Report.trainingFinished(network.getTrainer().getCurrentEpoch());

        // ── 3. Evaluation ────────────────────────
        // Checks how well the network works on the 20% of students it never saw.
        ClassifierEvaluator evaluator = new ClassifierEvaluator();
        evaluator.setThreshold(THRESHOLD);
        ClassificationMetrics metrics = evaluator.evaluate(network, testSet);
        ConfusionMatrix confusionMatrix = evaluator.getConfusionMatrix();
        Report.evaluation(confusionMatrix, metrics, THRESHOLD);

        // What the network learned: its two weights and its bias, per hour and per practice test.
        Report.weights(encoder.learnedWeights(network), RULE_OF_THE_DATA);

        // ── 4. Save, load and predict ────────────
        // Saves the network to models/exam-pass.dnet.
        // It can be used later without training again.
        Files.createDirectories(Path.of(MODEL).getParent());
        network.save(MODEL);

        // Loads the saved network and judges YourStudent, without training again.
        FeedForwardNetwork loadedNetwork = NeuralNetwork.load(MODEL, FeedForwardNetwork.class);
        PassPrediction prediction = new PassPredictor(loadedNetwork, encoder, THRESHOLD).predict(student);
        Report.prediction(student, prediction);
    }
}
