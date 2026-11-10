package com.deepnetts.examples.classification.exampass;

import com.deepnetts.examples.classification.exampass.dto.PassPrediction;
import com.deepnetts.examples.classification.exampass.dto.PassWeights;
import com.deepnetts.examples.classification.exampass.dto.Student;
import com.deepnetts.examples.classification.exampass.pipeline.StudentEncoder;

import deepnetts.eval.ClassificationMetrics;
import deepnetts.eval.ConfusionMatrix;

import java.util.Locale;

// Prints the output, so ExamPass contains only the workflow.
final class Report {

    // Every section line is this wide, for example "--- 1. Dataset -------...".
    private static final int WIDTH = 80;

    private Report() {
    }

    // Prints how many students there are and how many passed.
    static void dataset(int totalCount, int passedCount) {
        int failedCount = totalCount - passedCount;

        section("1. Dataset");
        System.out.printf(Locale.ROOT, "  Total:      %4d students%n", totalCount);
        System.out.printf(Locale.ROOT, "  Passed:     %4d (%.1f%%)%n", passedCount, 100.0 * passedCount / totalCount);
        System.out.printf(Locale.ROOT, "  Failed:     %4d (%.1f%%)%n", failedCount, 100.0 * failedCount / totalCount);
        System.out.printf(              "  Inputs:     %4d numbers per student (study hours, practice tests)%n", StudentEncoder.NUM_INPUTS);
        System.out.println();
    }

    // Prints when the training starts.
    static void trainingStarted() {
        section("2. Training");
        System.out.println("  Training the network...");
    }

    // Prints how many epochs the training took.
    static void trainingFinished(int epochs) {
        System.out.printf("  Stopped after %d epochs%n", epochs);
        System.out.println();
    }

    // Prints the results on the test set (students the network never saw).
    static void evaluation(ConfusionMatrix confusionMatrix, ClassificationMetrics metrics, float threshold) {
        int tp = confusionMatrix.getTruePositive();    // passed, and the network said pass
        int tn = confusionMatrix.getTrueNegative();    // failed, and the network said fail
        int fp = confusionMatrix.getFalsePositive();   // failed, but the network said pass
        int fn = confusionMatrix.getFalseNegative();   // passed, but the network said fail
        int total  = tp + tn + fp + fn;
        int baseline = Math.max(tp + fn, tn + fp);   // the accuracy of always answering with the more common answer

        section("3. Evaluation");
        System.out.println();
        System.out.printf(Locale.ROOT, "  Results on %d students the network was never trained on:%n", total);
        System.out.printf(Locale.ROOT, "  Threshold: %.2f - a probability at or above it counts as PASS%n", threshold);
        System.out.println();
        System.out.printf("  %-14s %16s %16s%n", "", "Predicted FAIL", "Predicted PASS");
        System.out.printf("  %-14s %11d (TN) %11d (FP)%n", "Actual FAIL", tn, fp);
        System.out.printf("  %-14s %11d (FN) %11d (TP)%n", "Actual PASS", fn, tp);
        System.out.println("  TN and TP are right answers, FP and FN are wrong ones.");
        System.out.println();
        System.out.printf(Locale.ROOT, "  Accuracy  %.1f%%  - %d of %d correct%n", 100 * metrics.getAccuracy(), tp + tn, total);
        System.out.printf(Locale.ROOT, "  Baseline  %.1f%%  - always answering with the more common answer would be this accurate%n",
                100.0 * baseline / total);
        System.out.println();
    }

    // Prints what the network learned about each input, next to the rule the data was made with.
    static void weights(PassWeights learned, PassWeights ofTheData) {
        section("4. What the network learned");
        System.out.println();
        System.out.println("  A network without a hidden layer adds up one weight per input. Read out of it:");
        System.out.println();
        System.out.printf(Locale.ROOT, "  %-18s %10s %13s %15s%n", "", "start", "per hour", "per test");
        System.out.printf(Locale.ROOT, "  %-18s %10.2f %13.3f %15.3f%n", "Learned:",
                learned.start(), learned.perStudyHour(), learned.perPracticeTest());
        System.out.printf(Locale.ROOT, "  %-18s %10.2f %13.3f %15.3f%n", "The data's rule:",
                ofTheData.start(), ofTheData.perStudyHour(), ofTheData.perPracticeTest());
        System.out.println();
        System.out.printf(Locale.ROOT, "  So one practice test counts as much as %.1f hours of study (the data's rule: %.1f).%n",
                learned.perPracticeTest() / learned.perStudyHour(), ofTheData.perPracticeTest() / ofTheData.perStudyHour());
        System.out.println();
    }

    // Prints your student and what the network says about them.
    static void prediction(Student student, PassPrediction prediction) {
        section("5. Your student");
        System.out.printf(Locale.ROOT, "  Study hours:     %.1f%n", student.studyHours());
        System.out.printf(              "  Practice tests:  %d%n", student.practiceTests());
        System.out.println();
        System.out.printf(Locale.ROOT, "  Pass probability:  %.1f%%%n", 100 * prediction.passProbability());
        System.out.printf(              "  Prediction:        %s%n", prediction.willPass() ? "PASS" : "FAIL");
        System.out.println();
        System.out.println(prediction.willPass()
                ? "  Result: The network expects your student to pass."
                : "  Result: The network expects your student to fail.");
        System.out.println();
        System.out.println("  Edit YourStudent.java and run again to explore how the result changes.");
    }

    // Prints a section title: "--- 1. Dataset " followed by dashes up to the full width.
    private static void section(String title) {
        String start = "--- " + title + " ";
        System.out.println(start + "-".repeat(Math.max(0, WIDTH - start.length())));
    }
}
