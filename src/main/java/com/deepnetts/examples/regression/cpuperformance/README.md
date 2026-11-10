# 10. CPU performance

**How fast is a CPU?**

**What you learn here:** Very uneven values (memory from 64 KB to 64 MB) go on a log scale, in the inputs and in the answer.

## Run it

From the root folder of the repository (the one with `pom.xml`, not this example's folder):

```bash
mvn clean compile exec:java -Pcpu-performance
```

Then change the input in [YourCpu.java](YourCpu.java) (cycle time, minimum and maximum memory, cache, minimum and maximum channels) and run it again.

## Result

Off by 38.7 points (baseline 125.4). R2 0.90; on 4 seeds 0.81-0.91.

Results are on test data the network never saw, with seed 42 (the one in the code); **on 4 seeds** shows the range with seeds 42, 1, 2 and 3. **Baseline** is the result without a network: always the most common answer, or always the average of the training data.

## Limitations

- The data is synthetic: it shows how the code works and how to read the results, not how well a network would do on real data.

## Data

[`datasets/cpu-performance.csv`](../../../../../../../../datasets/cpu-performance.csv): synthetic, made for this example: it shows how to use Deep Netts, not what real data would give.

## Files

Start reading in `CpuPerformance.java`: the whole workflow, from top to bottom.

| File | What it does |
|---|---|
| [`CpuPerformance.java`](CpuPerformance.java) | CPU performance example: a neural network learns from known CPUs. |
| [`Report.java`](Report.java) | Prints the output, so CpuPerformance contains only the workflow. |
| [`YourCpu.java`](YourCpu.java) | This is the CPU the network judges. |
| [`dto/Cpu.java`](dto/Cpu.java) | The input of the network: one CPU (computer processor) and its memory setup. |
| [`dto/PerformancePrediction.java`](dto/PerformancePrediction.java) | The output of the network: the performance score of a CPU. |
| [`pipeline/CpuEncoder.java`](pipeline/CpuEncoder.java) | Bridge between our records and the network, and between the CSV file and the network. |
| [`pipeline/CpuValidator.java`](pipeline/CpuValidator.java) | Checks that a CPU has valid values. |
| [`pipeline/FeatureExtractor.java`](pipeline/FeatureExtractor.java) | Turns a CPU into the numbers the network uses. |
| [`pipeline/PerformancePredictor.java`](pipeline/PerformancePredictor.java) | Predicts the performance score of a CPU. |

The interfaces `Predictor`, `ToRecord`, `ToTensor` and `CsvEncoder` are in [`util/`](../../util), shared by all examples.

---

Previous: [9. Home value](../bostonhousing) · [All examples](../../../../../../../../README.md#the-examples-in-a-suggested-order) · Next: [11. Auto insurance](../autoinsurance)
