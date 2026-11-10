# 11. Auto insurance

**What will a policy group cost per insured year?**

**What you learn here:** Some of the answer is chance: a small group has few claims, so its cost is partly luck, and no network can predict luck.

## Run it

From the root folder of the repository (the one with `pom.xml`, not this example's folder):

```bash
mvn clean compile exec:java -Pauto-insurance
```

Then change the input in [YourPolicyGroup.java](YourPolicyGroup.java) (kilometres class, bonus class, vehicle make, insured years, zone) and run it again.

## Result

Off by 0.50 thousand SEK (baseline 1.53). R2 0.85; on 4 seeds 0.81-0.86.

Results are on test data the network never saw, with seed 42 (the one in the code); **on 4 seeds** shows the range with seeds 42, 1, 2 and 3. **Baseline** is the result without a network: always the most common answer, or always the average of the training data.

## Limitations

- Part of the cost is chance, and no network can predict chance: a perfect R2 is not possible here.
- The data is synthetic: it shows how the code works and how to read the results, not how well a network would do on real data.

## Data

[`datasets/auto-insurance.csv`](../../../../../../../../datasets/auto-insurance.csv): synthetic, made for this example: it shows how to use Deep Netts, not what real data would give.

## Files

Start reading in `AutoInsurance.java`: the whole workflow, from top to bottom.

| File | What it does |
|---|---|
| [`AutoInsurance.java`](AutoInsurance.java) | Auto insurance example: a neural network learns from past policy groups. |
| [`Report.java`](Report.java) | Prints the output, so AutoInsurance contains only the workflow. |
| [`YourPolicyGroup.java`](YourPolicyGroup.java) | This is the policy group the network judges. |
| [`dto/PaymentPrediction.java`](dto/PaymentPrediction.java) | The output of the network: the expected insurance payment per insured year. |
| [`dto/PolicyGroup.java`](dto/PolicyGroup.java) | The input of the network: one group of insurance policies that share the same traits. |
| [`enums/Zone.java`](enums/Zone.java) | The zones the network knows (the area where the insured vehicles are mostly driven). |
| [`pipeline/FeatureExtractor.java`](pipeline/FeatureExtractor.java) | Turns a policy group into the numbers the network uses. |
| [`pipeline/PaymentPredictor.java`](pipeline/PaymentPredictor.java) | Predicts the expected insurance payment per insured year. |
| [`pipeline/PolicyEncoder.java`](pipeline/PolicyEncoder.java) | Bridge between our records and the network, and between the CSV file and the network. |
| [`pipeline/PolicyValidator.java`](pipeline/PolicyValidator.java) | Checks that a policy group has valid values. |

The interfaces `Predictor`, `ToRecord`, `ToTensor` and `CsvEncoder` are in [`util/`](../../util), shared by all examples.

---

Previous: [10. CPU performance](../cpuperformance) · [All examples](../../../../../../../../README.md#the-examples-in-a-suggested-order) · Next: [12. Sorting algorithm](../sortingalgorithm)
