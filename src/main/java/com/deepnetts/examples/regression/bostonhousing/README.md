# 9. Home value

**What is the typical home value in a neighborhood?**

**What you learn here:** Many inputs of different kinds (rates, counts, distances) in one regression.

## Run it

From the root folder of the repository (the one with `pom.xml`, not this example's folder):

```bash
mvn clean compile exec:java -Pboston-housing
```

Then change the input in [YourNeighborhood.java](YourNeighborhood.java) (crime rate, rooms, old buildings, distance to work, highway access, pupils per teacher, air pollution, lower-status share) and run it again.

## Result

Off by 2.5 thousand USD (baseline 7.2). R2 0.87; on 4 seeds 0.83-0.90.

Results are on test data the network never saw, with seed 42 (the one in the code); **on 4 seeds** shows the range with seeds 42, 1, 2 and 3. **Baseline** is the result without a network: always the most common answer, or always the average of the training data.

## Limitations

- The data is synthetic: it shows how the code works and how to read the results, not how well a network would do on real data.

## Data

[`datasets/boston-housing.csv`](../../../../../../../../datasets/boston-housing.csv): synthetic, made for this example: it shows how to use Deep Netts, not what real data would give.

## Files

Start reading in `BostonHousing.java`: the whole workflow, from top to bottom.

| File | What it does |
|---|---|
| [`BostonHousing.java`](BostonHousing.java) | Home value example: a neural network learns from known neighborhoods. |
| [`Report.java`](Report.java) | Prints the output, so BostonHousing contains only the workflow. |
| [`YourNeighborhood.java`](YourNeighborhood.java) | This is the neighborhood the network judges. |
| [`dto/HomeValuePrediction.java`](dto/HomeValuePrediction.java) | The output of the network: the typical home value in a neighborhood. |
| [`dto/Neighborhood.java`](dto/Neighborhood.java) | The input of the network: one neighborhood (a town or district in a city). |
| [`pipeline/FeatureExtractor.java`](pipeline/FeatureExtractor.java) | Turns a neighborhood into the numbers the network uses. |
| [`pipeline/HomeValuePredictor.java`](pipeline/HomeValuePredictor.java) | Predicts the typical home value in a neighborhood. |
| [`pipeline/NeighborhoodEncoder.java`](pipeline/NeighborhoodEncoder.java) | Bridge between our records and the network, and between the CSV file and the network. |
| [`pipeline/NeighborhoodValidator.java`](pipeline/NeighborhoodValidator.java) | Checks that a neighborhood has valid values. |

The interfaces `Predictor`, `ToRecord`, `ToTensor` and `CsvEncoder` are in [`util/`](../../util), shared by all examples.

---

Previous: [8. Iris flower](../../classification/irisflower) · [All examples](../../../../../../../../README.md#the-examples-in-a-suggested-order) · Next: [10. CPU performance](../cpuperformance)
