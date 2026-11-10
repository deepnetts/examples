# 8. Iris flower

**Which of three iris species is a flower?**

**What you learn here:** More than two answers: the target is one-hot (a 1 at the right species), the output layer uses SOFTMAX so the three probabilities add up to 1, and the test results are a 3 x 3 table instead of yes/no counts.

## Run it

From the root folder of the repository (the one with `pom.xml`, not this example's folder):

```bash
mvn clean compile exec:java -Piris-flower
```

Then change the input in [YourFlower.java](YourFlower.java) (sepal and petal length and width) and run it again.

## Result

100% correct on 29 test flowers (baseline 41.4%). One flower is 3.4 points; on 4 seeds 93-100%.

Results are on test data the network never saw, with seed 42 (the one in the code); **on 4 seeds** shows the range with seeds 42, 1, 2 and 3. **Baseline** is the result without a network: always the most common answer, or always the average.

## Limitations

- Only 29 test flowers, so one flower is 3.4 points. The three species are easy to tell apart; real classes rarely are.

## Data

[`datasets/iris-flowers.csv`](../../../../../../../../datasets/iris-flowers.csv): the classic [Iris](https://archive.ics.uci.edu/dataset/53/iris) data (UCI Machine Learning Repository, CC BY 4.0), 150 flowers without the 3 duplicate rows.

## Files

Start reading in `IrisFlower.java`: the whole workflow, from top to bottom.

| File | What it does |
|---|---|
| [`IrisFlower.java`](IrisFlower.java) | Iris flower example: a neural network learns from measured iris flowers. |
| [`Report.java`](Report.java) | Prints the output, so IrisFlower contains only the workflow. |
| [`YourFlower.java`](YourFlower.java) | This is the flower the network judges. |
| [`dto/Flower.java`](dto/Flower.java) | The input of the network: one iris flower, measured in centimeters. |
| [`dto/SpeciesMistake.java`](dto/SpeciesMistake.java) | One kind of mistake: a real species the network called another species, and how many times. |
| [`dto/SpeciesPrediction.java`](dto/SpeciesPrediction.java) | The output of the network for one flower: the probability of every species and the final answer. |
| [`enums/Species.java`](enums/Species.java) | The three iris species. The order is the order of the network's outputs. |
| [`pipeline/FeatureExtractor.java`](pipeline/FeatureExtractor.java) | Turns a flower into the numbers the network uses. |
| [`pipeline/FlowerEncoder.java`](pipeline/FlowerEncoder.java) | Bridge between our records and the network, in both directions, and between the CSV file and the network. |
| [`pipeline/FlowerValidator.java`](pipeline/FlowerValidator.java) | Checks that a flower has valid values. |
| [`pipeline/SpeciesConfusion.java`](pipeline/SpeciesConfusion.java) | Counts, for every real species, which species the network said. |
| [`pipeline/SpeciesPredictor.java`](pipeline/SpeciesPredictor.java) | Predicts the species of one flower. |

The interfaces `Predictor`, `ToRecord`, `ToTensor` and `CsvEncoder` are in [`util/`](../../util), shared by all examples.

---

Previous: [7. Sonar mine](../sonarmines) · [All examples](../../../../../../../../README.md#the-examples-in-a-suggested-order) · Next: [9. Home value](../../regression/bostonhousing)
