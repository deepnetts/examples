package com.deepnetts.examples.regression.taxifare;

import com.deepnetts.examples.regression.taxifare.dto.FareFormula;
import com.deepnetts.examples.regression.taxifare.dto.FarePrediction;
import com.deepnetts.examples.regression.taxifare.dto.Ride;
import com.deepnetts.examples.regression.taxifare.pipeline.RideEncoder;

import java.util.Locale;

// Prints the output, so TaxiFare contains only the workflow.
final class Report {

    // Every section line is this wide, for example "--- 1. Dataset -------...".
    private static final int WIDTH = 80;

    private Report() {
    }

    // Prints how many rides there are.
    static void dataset(int totalCount) {
        section("1. Dataset");
        System.out.printf(Locale.ROOT, "  Total:      %4d taxi rides%n", totalCount);
        System.out.printf(              "  Inputs:     %4d numbers per ride (distance in km, duration in minutes)%n", RideEncoder.NUM_INPUTS);
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

    // Prints the results on the test set (rides the network never saw).
    static void evaluation(int testCount, double averageErrorEur, double baselineErrorEur, double r2) {
        section("3. Evaluation");
        System.out.println();
        System.out.printf(Locale.ROOT, "  Results on %d rides the network was never trained on:%n", testCount);
        System.out.println();
        System.out.printf(Locale.ROOT, "  Average error  %5.2f EUR  - on average the predicted fare is off by this much%n", averageErrorEur);
        System.out.printf(Locale.ROOT, "  Baseline       %5.2f EUR  - always answering with the average fare would be off by this much%n", baselineErrorEur);
        System.out.printf(Locale.ROOT, "  R2             %5.3f      - share of the differences between fares explained (1.0 = all)%n", r2);
        System.out.println();
    }

    // Prints the formula the network learned, next to the formula the data was made with.
    static void formula(FareFormula learned, FareFormula ofTheData) {
        section("4. What the network learned");
        System.out.println();
        System.out.println("  A network without a hidden layer is a formula. Read out of its weights:");
        System.out.println();
        System.out.printf(Locale.ROOT, "  Learned:          fare = %.2f + %.2f per km + %.2f per minute%n",
                learned.startPriceEur(), learned.pricePerKmEur(), learned.pricePerMinuteEur());
        System.out.printf(Locale.ROOT, "  The data's rule:  fare = %.2f + %.2f per km + %.2f per minute (plus a little noise)%n",
                ofTheData.startPriceEur(), ofTheData.pricePerKmEur(), ofTheData.pricePerMinuteEur());
        System.out.println();
    }

    // Prints your ride and the fare the network expects.
    static void prediction(Ride ride, FarePrediction prediction) {
        section("5. Your ride");
        System.out.printf(Locale.ROOT, "  Distance:  %.1f km%n", ride.distanceKm());
        System.out.printf(Locale.ROOT, "  Duration:  %.0f minutes%n", ride.durationMinutes());
        System.out.println();
        System.out.printf(Locale.ROOT, "  Result: The network expects a fare of about %.2f EUR.%n", prediction.fareEur());
        System.out.println();
        System.out.println("  Edit YourRide.java and run again to explore how the result changes.");
    }

    // Prints a section title: "--- 1. Dataset " followed by dashes up to the full width.
    private static void section(String title) {
        String start = "--- " + title + " ";
        System.out.println(start + "-".repeat(Math.max(0, WIDTH - start.length())));
    }
}
