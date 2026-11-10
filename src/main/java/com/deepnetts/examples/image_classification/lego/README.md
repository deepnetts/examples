# 15. Lego figure

**Does a picture show a Lego figure?**

**What you learn here:** Choosing the other class: the "not Lego" pictures include Duke drawings, which have the same white background, so the network can't decide by the background alone.

## Run it

From the root folder of the repository (the one with `pom.xml`, not this example's folder):

```bash
mvn clean compile exec:java -Plego-figure
```

It takes about 3 minutes.

Then change the input in [YourImage.java](YourImage.java) (the path of an image file) and run it again.

## Result

96.7% correct on 182 test pictures (baseline 53.8%). On 4 seeds 97-98%. Probably easier than real use: the other pictures come from other datasets, so part of the difference is the kind of picture, not only the figure.

Results are on test data the network never saw, with seed 42 (the one in the code); **on 4 seeds** shows the range with seeds 42, 1, 2 and 3. **Baseline** is the result without a network: always the most common answer, or always the average.

## Limitations

- The other pictures come from other datasets (Duke, food, parking), so part of what the network learns is the kind of picture, not only the figure. Pictures of other toys would be a harder and more honest test.

## Data

[`datasets/LegoSet`](../../../../../../../../datasets/LegoSet): 456 pictures of Lego figures from the Deep Netts cookbook, and 456 other pictures from this project: 49 Duke drawings, 300 food photos and 107 parking spots, all 96 x 96.

## Files

Start reading in `LegoFigure.java`: the whole workflow, from top to bottom.

| File | What it does |
|---|---|
| [`LegoFigure.java`](LegoFigure.java) | Lego example: a convolutional neural network learns from pictures of Lego figures and of other things |
| [`Report.java`](Report.java) | Prints the output, so LegoFigure contains only the workflow. |
| [`YourImage.java`](YourImage.java) | This is the picture the network judges. |
| [`dto/LegoPrediction.java`](dto/LegoPrediction.java) | The output of the network: how sure it is that the picture shows a Lego figure. |
| [`pipeline/LegoEncoder.java`](pipeline/LegoEncoder.java) | Bridge between images and the network, and between the image folder and the network. |
| [`pipeline/LegoFileReader.java`](pipeline/LegoFileReader.java) | Reads one image from a file and checks that it is really an image. |
| [`pipeline/LegoPredictor.java`](pipeline/LegoPredictor.java) | Predicts if a picture shows a Lego figure. |

The interfaces `Predictor`, `ToRecord`, `ToTensor` and `CsvEncoder` are in [`util/`](../../util), shared by all examples.

---

Previous: [14. Duke detection](../dukedetection) · [All examples](../../../../../../../../README.md#the-examples-in-a-suggested-order) · Next: [16. Parking spot](../parking)
