# 17. Hot dog

**Does a photo show a hot dog?**

**What you learn here:** The limit of a small network: real food photos are all different (light, angle, background), and 300 training photos are too few to learn that from scratch. And what really helps instead: a pretrained network (next example).

## Run it

From the root folder of the repository (the one with `pom.xml`, not this example's folder):

```bash
mvn clean compile exec:java -Phot-dog
```

It takes about 3 minutes.

Then change the input in [YourImage.java](YourImage.java) (the path of an image file) and run it again.

## Result

61.3% correct on 93 test photos (baseline 52.7%) with seed 42, but 47-54% on seeds 1, 2 and 3, below their baselines. So the network does not really learn what a hot dog looks like: 300 photos this different are too few for a small network trained from scratch. Bigger images, Adam with batches (as in the cookbook) and mirrored copies of the photos were all tried; none helped. The program says so itself: when the result is not far above the baseline, it prints a note under the numbers. A pretrained network is the way out: VGG16 (the next example) was trained on 1.2 million photos and knows `hotdog` as one of its 1000 categories.

Results are on test data the network never saw, with seed 42 (the one in the code); **on 4 seeds** shows the range with seeds 42, 1, 2 and 3. **Baseline** is the result without a network: always the most common answer, or always the average.

## Limitations

- The network does not really learn (see the result). The example is here to show that limit: use a pretrained network (VGG16 import) or many more photos.

## Data

[`datasets/HotDogSet`](../../../../../../../../datasets/HotDogSet): 467 photos of hot dogs and of other food from the Deep Netts cookbook, made smaller (at most 128 pixels).

## Files

Start reading in `HotDog.java`: the whole workflow, from top to bottom.

| File | What it does |
|---|---|
| [`HotDog.java`](HotDog.java) | Hot dog example: a convolutional neural network learns from photos of hot dogs and of other food. |
| [`Report.java`](Report.java) | Prints the output, so HotDog contains only the workflow. |
| [`YourImage.java`](YourImage.java) | This is the photo the network judges. |
| [`dto/HotDogPrediction.java`](dto/HotDogPrediction.java) | The output of the network: how sure it is that the photo shows a hot dog. |
| [`pipeline/HotDogEncoder.java`](pipeline/HotDogEncoder.java) | Bridge between images and the network, and between the image folder and the network. |
| [`pipeline/HotDogFileReader.java`](pipeline/HotDogFileReader.java) | Reads one image from a file and checks that it is really an image. |
| [`pipeline/HotDogPredictor.java`](pipeline/HotDogPredictor.java) | Predicts if a photo shows a hot dog. |

The interfaces `Predictor`, `ToRecord`, `ToTensor` and `CsvEncoder` are in [`util/`](../../util), shared by all examples.

---

Previous: [16. Parking spot](../parking) · [All examples](../../../../../../../../README.md#the-examples-in-a-suggested-order) · Next: [18. VGG16 import](../vgg16)
