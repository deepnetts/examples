# 7. Sonar mine

**Is an object on the seabed a mine?**

**What you learn here:** A category input (the seabed type) becomes several 0/1 inputs, one per category (one-hot), so the network doesn't read an order into it.

## Run it

From the root folder of the repository (the one with `pom.xml`, not this example's folder):

```bash
mvn clean compile exec:java -Psonar-mines
```

Then change the input in [YourContact.java](YourContact.java) (range, aspect angle, echo level and duration, spectral centroid, shadow length, seabed type) and run it again.

## Result

76.6% correct (baseline 58.7%), finds 72% of mines. On 4 seeds: 76-78% correct.

Results are on test data the network never saw, with seed 42 (the one in the code); **on 4 seeds** shows the range with seeds 42, 1, 2 and 3. **Baseline** is the result without a network: always the most common answer, or always the average.

## Limitations

- The data is synthetic: it shows how the code works and how to read the results, not how well a network would do on real data.

## Data

[`datasets/sonar-mines.csv`](../../../../../../../../datasets/sonar-mines.csv): synthetic, made for this example: it shows how to use Deep Netts, not what real data would give.

## Files

Start reading in `SonarMines.java`: the whole workflow, from top to bottom.

| File | What it does |
|---|---|
| [`Report.java`](Report.java) | Prints the output, so SonarMines contains only the workflow. |
| [`SonarMines.java`](SonarMines.java) | Sonar mine detection example: a neural network learns from past sonar contacts. |
| [`YourContact.java`](YourContact.java) | This is the sonar contact the network judges. |
| [`dto/MinePrediction.java`](dto/MinePrediction.java) | The output of the network: mine probability and the final prediction. |
| [`dto/SonarContact.java`](dto/SonarContact.java) | The input of the network: one sonar contact, an object on the seabed that sent back an echo. |
| [`enums/SeabedType.java`](enums/SeabedType.java) | The seabed types the network knows. |
| [`pipeline/ContactEncoder.java`](pipeline/ContactEncoder.java) | Bridge between our records and the network, and between the CSV file and the network. |
| [`pipeline/ContactValidator.java`](pipeline/ContactValidator.java) | Checks that a sonar contact has valid values. |
| [`pipeline/FeatureExtractor.java`](pipeline/FeatureExtractor.java) | Turns a sonar contact into the numbers the network uses. |
| [`pipeline/MinePredictor.java`](pipeline/MinePredictor.java) | Predicts whether one sonar contact is a mine. |

The interfaces `Predictor`, `ToRecord`, `ToTensor` and `CsvEncoder` are in [`util/`](../../util), shared by all examples.

---

Previous: [6. Spam email](../spamemail) · [All examples](../../../../../../../../README.md#the-examples-in-a-suggested-order) · Next: [8. Iris flower](../irisflower)
