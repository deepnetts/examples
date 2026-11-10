# 18. VGG16 import

**What does an image show, out of 1000 everyday things?**

**What you learn here:** Using a network trained somewhere else: VGG16 was trained in TensorFlow on 1.2 million ImageNet photos. Deep Netts builds the same layers, copies the weights in and runs it in pure Java, without training.

## Run it

From the root folder of the repository (the one with `pom.xml`, not this example's folder):

Give it 4 GB of memory. On Windows PowerShell:

```powershell
$env:MAVEN_OPTS="-Xmx4g"; mvn clean compile exec:java -Pvgg16-import
```

On Windows cmd:

```bat
set "MAVEN_OPTS=-Xmx4g" && mvn clean compile exec:java -Pvgg16-import
```

On Mac and Linux:

```bash
MAVEN_OPTS="-Xmx4g" mvn clean compile exec:java -Pvgg16-import
```

It takes about 5 minutes on a laptop (building the network and reading the weights about 4, then about a minute per image).

The first run downloads the VGG16 weights (about 600 MB) into `models/cache`; later runs reuse them. The network needs about 4 GB of free memory (`-Xmx4g`). On a computer with 8 GB of RAM, close the browser and other programs first: otherwise the system moves memory to disk and the run can take an hour. Deep Netts can import the weights with `TensorflowUtils.importWeights(network, file)` in one call, but for VGG16 that needs about 8 GB; `Vgg16WeightsReader` reads the same file number by number instead.

Then change the input in [YourImage.java](YourImage.java) (the path of an image file) and run it again.

## Result

For `elephant.jpg` the best answer is `African_elephant` (72.9%), for `tiger.jpg` it is `tiger` (73.5%). There is no test set: VGG16 is used as it is, and its own test results on ImageNet are published by its authors (about 90% of the time the right answer is among its best 5).

## Limitations

- It only uses the network as it is: no training on your own pictures (fine-tuning). It knows only the 1000 ImageNet categories, and it sees every image as 224 x 224, so small photos lose detail. It needs about 4 GB of free memory and a few minutes on a laptop CPU.

## Data

The 1000 ImageNet category names and a few test images; the weights (about 600 MB) are downloaded on the first run into `models/cache`.

## Files

Start reading in `Vgg16Import.java`: the whole workflow, from top to bottom.

| File | What it does |
|---|---|
| [`Report.java`](Report.java) | Prints the output, so Vgg16Import contains only the workflow. |
| [`Vgg16Import.java`](Vgg16Import.java) | VGG16 import example: uses a network that was trained somewhere else. |
| [`YourImage.java`](YourImage.java) | This is the image VGG16 judges. |
| [`dto/LabelScore.java`](dto/LabelScore.java) | One of the 1000 ImageNet categories and how sure the network is that the image shows it. |
| [`dto/Vgg16Prediction.java`](dto/Vgg16Prediction.java) | The output of VGG16 for one image: its best answers, the most likely first. |
| [`pipeline/ImageNetLabels.java`](pipeline/ImageNetLabels.java) | The names of the 1000 ImageNet categories, in the order of VGG16's outputs. |
| [`pipeline/Vgg16Encoder.java`](pipeline/Vgg16Encoder.java) | Bridge between images and VGG16. |
| [`pipeline/Vgg16Network.java`](pipeline/Vgg16Network.java) | Builds the empty VGG16 network: the same layers, in the same order, as the network that was trained in TensorFlow. |
| [`pipeline/Vgg16Predictor.java`](pipeline/Vgg16Predictor.java) | Asks VGG16 what an image shows and keeps its best answers. |
| [`pipeline/Vgg16Weights.java`](pipeline/Vgg16Weights.java) | Gets the VGG16 weights that were trained in TensorFlow: downloads them once (about 600 MB), then reuses the file. |
| [`pipeline/Vgg16WeightsReader.java`](pipeline/Vgg16WeightsReader.java) | Copies the weights exported from TensorFlow into the VGG16 network built by Vgg16Network. |

The interfaces `Predictor`, `ToRecord`, `ToTensor` and `CsvEncoder` are in [`util/`](../../util), shared by all examples.

---

Previous: [17. Hot dog](../hotdog) · [All examples](../../../../../../../../README.md#the-examples-in-a-suggested-order)
