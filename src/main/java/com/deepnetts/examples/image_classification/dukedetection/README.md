# 14. Duke detection

**Does an image show Duke, the Java mascot?**

**What you learn here:** The first convolutional network: filters that slide over the image, max pooling, then the usual layers. Images instead of a CSV.

## Run it

From the root folder of the repository (the one with `pom.xml`, not this example's folder):

```bash
mvn clean compile exec:java -Pduke-detection
```

It takes about 30 seconds.

Then change the input in [YourImage.java](YourImage.java) (the path of an image file) and run it again.

## Result

94.7% correct on 19 test images (baseline 73.7%). One image is 5.3 points; on 4 seeds 89-100%.

Results are on test data the network never saw, with seed 42 (the one in the code); **on 4 seeds** shows the range with seeds 42, 1, 2 and 3. **Baseline** is the result without a network: always the most common answer, or always the average.

## Limitations

- Only 19 test images, so one image is 5.3 points. The other pictures come from other datasets (Lego, food, parking), which makes them easier to tell apart from Duke.

## Data

[`datasets/DukeSet`](../../../../../../../../datasets/DukeSet): 49 drawings of Duke and, as the other class, 49 pictures from this project's other datasets: 20 Lego figures, 15 food photos and 14 parking spots.

## Files

Start reading in `DukeDetection.java`: the whole workflow, from top to bottom.

| File | What it does |
|---|---|
| [`DukeDetection.java`](DukeDetection.java) | Duke detection example: a convolutional neural network learns from images of Duke (the Java mascot) |
| [`Report.java`](Report.java) | Prints the output, so DukeDetection contains only the workflow. |
| [`YourImage.java`](YourImage.java) | This is the image the network judges. |
| [`dto/DukePrediction.java`](dto/DukePrediction.java) | The output of the network: how sure it is that the image shows Duke. |
| [`pipeline/DukeEncoder.java`](pipeline/DukeEncoder.java) | Bridge between images and the network, and between the image folder and the network. |
| [`pipeline/DukeFileReader.java`](pipeline/DukeFileReader.java) | Reads one image from a file and checks that it is really an image. |
| [`pipeline/DukePredictor.java`](pipeline/DukePredictor.java) | Predicts if an image shows Duke. |

The interfaces `Predictor`, `ToRecord`, `ToTensor` and `CsvEncoder` are in [`util/`](../../util), shared by all examples.

---

Previous: [13. Handwritten digits](../digits) · [All examples](../../../../../../../../README.md#the-examples-in-a-suggested-order) · Next: [15. Lego figure](../lego)
