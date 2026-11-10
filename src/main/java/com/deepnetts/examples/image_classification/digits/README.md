# 13. Handwritten digits

**Which digit, 0 to 9, is in a small gray image?**

**What you learn here:** Ten answers from images: gray images (one value per pixel instead of three), SOFTMAX over 10 outputs and a 10 x 10 table of mistakes.

## Run it

From the root folder of the repository (the one with `pom.xml`, not this example's folder):

```bash
mvn clean compile exec:java -Pdigits
```

It takes about 40 seconds.

Then change the input in [YourImage.java](YourImage.java) (the path of an image file) and run it again.

## Result

91.5% correct on 200 test images (baseline 11.5%). On 4 seeds 88-92%. 9 and 5 are read correctly least often.

Results are on test data the network never saw, with seed 42 (the one in the code); **on 4 seeds** shows the range with seeds 42, 1, 2 and 3. **Baseline** is the result without a network: always the most common answer, or always the average.

## Limitations

- 1000 of the 70,000 MNIST digits. Your digit must look like them: light on dark, centered, filling most of the image.

## Data

[`datasets/DigitSet`](../../../../../../../../datasets/DigitSet): 1000 images (100 of each digit) from [MNIST](https://yann.lecun.com/exdb/mnist/), the classic set of handwritten digits.

## Files

Start reading in `Digits.java`: the whole workflow, from top to bottom.

| File | What it does |
|---|---|
| [`Digits.java`](Digits.java) | Digits example: a convolutional neural network learns to read handwritten digits, 0 to 9. |
| [`Report.java`](Report.java) | Prints the output, so Digits contains only the workflow. |
| [`YourImage.java`](YourImage.java) | This is the image the network reads. |
| [`dto/DigitMistake.java`](dto/DigitMistake.java) | One kind of mistake: a real digit the network read as another digit, and how many times. |
| [`dto/DigitPrediction.java`](dto/DigitPrediction.java) | The output of the network for one image: which digit it sees and how sure it is. |
| [`pipeline/DigitConfusion.java`](pipeline/DigitConfusion.java) | Counts, for every real digit, which digit the network said. |
| [`pipeline/DigitEncoder.java`](pipeline/DigitEncoder.java) | Bridge between images and the network, in both directions, and between the image folder and the network. |
| [`pipeline/DigitFileReader.java`](pipeline/DigitFileReader.java) | Reads one image from a file and checks that it is really an image. |
| [`pipeline/DigitPredictor.java`](pipeline/DigitPredictor.java) | Predicts which digit an image shows. |

The interfaces `Predictor`, `ToRecord`, `ToTensor` and `CsvEncoder` are in [`util/`](../../util), shared by all examples.

---

Previous: [12. Sorting algorithm](../../regression/sortingalgorithm) · [All examples](../../../../../../../../README.md#the-examples-in-a-suggested-order) · Next: [14. Duke detection](../dukedetection)
