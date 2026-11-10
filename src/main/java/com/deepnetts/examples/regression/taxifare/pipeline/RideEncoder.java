package com.deepnetts.examples.regression.taxifare.pipeline;

import com.deepnetts.examples.regression.taxifare.dto.FareFormula;
import com.deepnetts.examples.regression.taxifare.dto.Ride;
import com.deepnetts.examples.util.CsvEncoder;

import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.layers.OutputLayer;
import deepnetts.tensor.Tensor1D;
import deepnetts.tensor.Tensor2D;

import java.util.List;

// Turns data into the numbers the network learns from:
// a CSV row becomes a Ride (toRecord), a Ride becomes the network inputs (toTensor),
// and the fare column becomes the correct answer (toTarget). load() (from CsvEncoder) reads the whole CSV this way.
public final class RideEncoder implements CsvEncoder<Ride> {

    // The expected CSV header: the two ride fields and the fare (the value to predict).
    private static final List<String> HEADER = List.of("distanceKm", "durationMinutes", "fareEur");

    private static final int FARE_COLUMN = HEADER.size() - 1;

    // The largest fare in the data (rounded up). The fare is divided by it, so the network also works with 0 to 1.
    private static final double MAX_FARE_EUR = 70;

    public static final int NUM_INPUTS = FeatureExtractor.NUM_FEATURES;
    public static final int NUM_OUTPUTS = 1;   // one output: the fare

    @Override
    public List<String> header() {
        return HEADER;
    }

    @Override
    public List<String> inputNames() {
        return FeatureExtractor.FEATURE_NAMES;
    }

    @Override
    public List<String> outputNames() {
        return List.of(HEADER.get(FARE_COLUMN));
    }

    // Converts a ride into the numbers the network uses (a tensor: Deep Netts' array of numbers).
    @Override
    public Tensor1D toTensor(Ride ride) {
        RideValidator.validate(ride);
        return Tensor1D.of(FeatureExtractor.extract(ride));
    }

    // Converts a CSV row into a ride.
    // CSV columns: distanceKm, durationMinutes.
    @Override
    public Ride toRecord(String[] columns) {
        return new Ride(Double.parseDouble(columns[0]), Double.parseDouble(columns[1]));
    }

    // The correct answer of a CSV row: the fare, on the 0 to 1 scale the network works with.
    @Override
    public Tensor1D toTarget(String[] columns) {
        double fare = Double.parseDouble(columns[FARE_COLUMN]);
        if (!Double.isFinite(fare) || fare < 0) {
            throw new IllegalArgumentException("fareEur must be a number >= 0, was " + columns[FARE_COLUMN]);
        }
        return Tensor1D.of(new float[] {(float) (fare / MAX_FARE_EUR)});
    }

    // The network's answer (0 to 1) back in EUR.
    public double toFareEur(float networkOutput) {
        return networkOutput * MAX_FARE_EUR;
    }

    // Reads the formula the network learned.
    // A network without a hidden layer computes: output = bias + weight1 * input1 + weight2 * input2.
    // The inputs and the output were divided by their largest values, so the weights are turned back into EUR here.
    public FareFormula learnedFormula(FeedForwardNetwork network) {
        OutputLayer output = (OutputLayer) network.getLayers().getLast();
        Tensor2D weights = output.getWeights();
        double bias = output.getBiases().get(0);
        return new FareFormula(
                bias * MAX_FARE_EUR,
                weights.get(0, 0) * MAX_FARE_EUR / FeatureExtractor.MAX_DISTANCE_KM,
                weights.get(0, 1) * MAX_FARE_EUR / FeatureExtractor.MAX_DURATION_MINUTES);
    }
}
