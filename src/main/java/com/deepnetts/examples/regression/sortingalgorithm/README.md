# 12. Sorting algorithm

**Which of 9 sorting algorithms is the fastest for an array?**

**What you learn here:** Predict a number for every option, then choose: the network predicts each algorithm's time and the fastest is picked. The times were measured on a real computer, and they go from 0.1 microseconds to 38 ms, so the network learns their log.

## Run it

From the root folder of the repository (the one with `pom.xml`, not this example's folder):

```bash
mvn clean compile exec:java -Psorting-algorithm
```

Then change the input in [YourArray.java](YourArray.java) (algorithm, array size, share already sorted, numbers out of place) and run it again.

## Result

Times off by 18% for half of the test runs. Picked the fastest on 64 of 100 new arrays (always Quick Sort: 46), and one at most 10% slower than the fastest on 73 (always Quick Sort: 53). Sorting all of them took 16% longer than the best possible (always Quick Sort: 43%). On 4 seeds: 64-80 fastest, 73-89 within 10%, 3-16% longer.

Results are on test data the network never saw, with seed 42 (the one in the code); **on 4 seeds** shows the range with seeds 42, 1, 2 and 3. **Baseline** is the result without a network: always the most common answer, or always the average.

## Limitations

- The times were measured on one laptop. Another computer (or Java version) gives other times: run the benchmark to measure yours.

To measure your own computer (under a minute; close other programs first), run the benchmark, then set `DATASET` in `SortingAlgorithm.java` to `datasets/sorting-algorithm-my-computer.csv`:

```bash
mvn clean compile exec:java -Psorting-benchmark
```

## Data

[`datasets/sorting-algorithm.csv`](../../../../../../../../datasets/sorting-algorithm.csv): real times, measured by `SortingBenchmark` on one laptop (Intel Core i5-1135G7, Java 25).

## Files

Start reading in `SortingAlgorithm.java`: the whole workflow, from top to bottom.

| File | What it does |
|---|---|
| [`Report.java`](Report.java) | Prints the output, so SortingAlgorithm contains only the workflow. |
| [`SortingAlgorithm.java`](SortingAlgorithm.java) | Sorting algorithm example (regression): a neural network learns how long 9 sorting algorithms take, |
| [`YourArray.java`](YourArray.java) | This is the array the network ranks the algorithms for. |
| [`benchmark/SortingBenchmark.java`](benchmark/SortingBenchmark.java) | Makes the dataset of the SortingAlgorithm example: it sorts arrays with all 9 algorithms and measures how long it takes. |
| [`benchmark/Sorts.java`](benchmark/Sorts.java) | The 9 sorting algorithms that SortingBenchmark measures. Each one sorts an int array in place, from small to large. |
| [`dto/ArrayToSort.java`](dto/ArrayToSort.java) | An array you want to sort: the network says how long each algorithm would take. |
| [`dto/ChoiceResult.java`](dto/ChoiceResult.java) | How good the network's choice of algorithm is on the test arrays, next to always using one algorithm. |
| [`dto/MeasuredRun.java`](dto/MeasuredRun.java) | One row of the CSV file: a sort run and how long it took. |
| [`dto/SortRun.java`](dto/SortRun.java) | The input of the network: one algorithm sorting one array. |
| [`dto/SortingAdvice.java`](dto/SortingAdvice.java) | The predicted time of every algorithm for one array, fastest first. |
| [`dto/TimeAccuracy.java`](dto/TimeAccuracy.java) | How close the predicted times are to the measured ones. |
| [`dto/TimePrediction.java`](dto/TimePrediction.java) | The output of the network for one algorithm: how long it would take to sort the array. |
| [`enums/Algorithm.java`](enums/Algorithm.java) | The sorting algorithms in the data, by the name used in the CSV file. |
| [`pipeline/FeatureExtractor.java`](pipeline/FeatureExtractor.java) | Turns a sort run into the numbers the network uses. |
| [`pipeline/RunEncoder.java`](pipeline/RunEncoder.java) | Bridge between our records and the network, and between the CSV file and the network. |
| [`pipeline/RunValidator.java`](pipeline/RunValidator.java) | Checks that an array and a sort run have valid values. |
| [`pipeline/SortingAdvisor.java`](pipeline/SortingAdvisor.java) | Predicts how long every algorithm would take to sort an array, and puts the fastest first. |
| [`pipeline/SortingEvaluator.java`](pipeline/SortingEvaluator.java) | Calculates the results that Report prints: how often each algorithm is the fastest, |

The interfaces `Predictor`, `ToRecord`, `ToTensor` and `CsvEncoder` are in [`util/`](../../util), shared by all examples.

---

Previous: [11. Auto insurance](../autoinsurance) · [All examples](../../../../../../../../README.md#the-examples-in-a-suggested-order) · Next: [13. Handwritten digits](../../image_classification/digits)
