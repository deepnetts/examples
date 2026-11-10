# Deep Netts examples

Runnable examples for the [Deep Netts](https://www.deepnetts.com) Community Edition (`deepnetts-core` 4.0.5).
Each example is a small program that learns something from data, tests itself, and then judges an input that you can change.
You read the code from top to bottom, run it, change the input in its `Your...java` file and run it again.

This page shows everything you need to run them, in a suggested order. The order is only a suggestion: every example works on its own.

## What you need

- **Git** (to download the code; you can also download a ZIP from GitHub)
- **Java 25 or newer** (a JDK, not only a JRE)
- **Maven 3.9 or newer** ([how to install Maven](https://maven.apache.org/install.html))

Check what you have. Each command prints a version:

```bash
git --version
```

```bash
java --version
```

```bash
javac --version
```

```bash
mvn --version
```

`java` and `javac` must say 25 or higher. `mvn --version` also shows the Java that Maven uses: that one must be 25 or higher too.
If it is older, install a JDK 25 (for example from [Adoptium](https://adoptium.net)) and point `JAVA_HOME` to it.

## Get the code

```bash
git clone https://github.com/deepnetts/examples.git
```

```bash
cd examples
```

Stay in this folder: it is the one with `pom.xml`. All commands below are run from here.

## Run your first example

```bash
mvn clean compile exec:java
```

Without anything else, this runs Hello World: under a second of training that checks that Java and Deep Netts work.
The first run takes longer: Maven downloads Deep Netts and its plugins (about 20 MB). You should see:

```
--- 1. Setup -------------------------------------------------------------------
  Java version: 25

--- 2. Training ----------------------------------------------------------------
  Training the network...
  Stopped after 23 epochs

--- 3. Your temperature --------------------------------------------------------
  37 C is 98.6 F, the network says 98.6 F

  SUCCESS: Java and Deep Netts work. Go on with the next example in the main README.
```

If you see `SUCCESS`, everything is ready.

## Run any example

Add `-P` and the name of the example (the name is in the table below):

```bash
mvn clean compile exec:java -Pspam-email
```

Every example prints its steps: the data, the training, the test on data it never saw, and the answer for the input in its `Your...java` file
(for example `YourEmail.java`). Open that file, change the numbers, and run the same command again to see how the answer changes.
Each example has its own README next to its code, with the expected output and the files to read.

## The examples in a suggested order

You can start anywhere. This order goes from the whole workflow, through the main ideas, to images and a ready-made network.

| # | Example | Takes | Command |
|---|---|---|---|
| 1 | [Hello World](src/main/java/com/deepnetts/examples/helloworld/README.md) | about 10 seconds | `-Phello-world` |
| 2 | [Credit-card fraud](src/main/java/com/deepnetts/examples/classification/creditcardfraud/README.md) | about 10 seconds | `-Pcredit-card-fraud` |
| 3 | [Taxi fare (linear regression)](src/main/java/com/deepnetts/examples/regression/taxifare/README.md) | about 10 seconds | `-Ptaxi-fare` |
| 4 | [Exam pass (logistic regression)](src/main/java/com/deepnetts/examples/classification/exampass/README.md) | about 10 seconds | `-Pexam-pass` |
| 5 | [Light switches (XOR)](src/main/java/com/deepnetts/examples/classification/lightswitch/README.md) | about 10 seconds | `-Plight-switches` |
| 6 | [Spam email](src/main/java/com/deepnetts/examples/classification/spamemail/README.md) | about 10 seconds | `-Pspam-email` |
| 7 | [Sonar mine](src/main/java/com/deepnetts/examples/classification/sonarmines/README.md) | about 10 seconds | `-Psonar-mines` |
| 8 | [Iris flower](src/main/java/com/deepnetts/examples/classification/irisflower/README.md) | about 10 seconds | `-Piris-flower` |
| 9 | [Home value](src/main/java/com/deepnetts/examples/regression/bostonhousing/README.md) | about 10 seconds | `-Pboston-housing` |
| 10 | [CPU performance](src/main/java/com/deepnetts/examples/regression/cpuperformance/README.md) | about 10 seconds | `-Pcpu-performance` |
| 11 | [Auto insurance](src/main/java/com/deepnetts/examples/regression/autoinsurance/README.md) | about 20 seconds | `-Pauto-insurance` |
| 12 | [Sorting algorithm](src/main/java/com/deepnetts/examples/regression/sortingalgorithm/README.md) | about 20 seconds | `-Psorting-algorithm` |
| 13 | [Handwritten digits](src/main/java/com/deepnetts/examples/image_classification/digits/README.md) | about 40 seconds | `-Pdigits` |
| 14 | [Duke detection](src/main/java/com/deepnetts/examples/image_classification/dukedetection/README.md) | about 20 seconds | `-Pduke-detection` |
| 15 | [Lego figure](src/main/java/com/deepnetts/examples/image_classification/lego/README.md) | about 3 minutes | `-Plego-figure` |
| 16 | [Parking spot](src/main/java/com/deepnetts/examples/image_classification/parking/README.md) | about 40 seconds | `-Pparking-spot` |
| 17 | [Hot dog](src/main/java/com/deepnetts/examples/image_classification/hotdog/README.md) | about 3 minutes | `-Phot-dog` |
| 18 | [VGG16 import](src/main/java/com/deepnetts/examples/image_classification/vgg16/README.md) | about 5 minutes | `-Pvgg16-import` |

Times were measured on a laptop for the whole command (`mvn clean compile exec:java`), after Maven had downloaded everything.

### 1. Hello World

Check that Java and Deep Netts work, in under a second. Run it first.

**You learn:** That the setup works, and the smallest training loop: make data, build a network, train it, ask it.

**Takes:** about 10 seconds.

```bash
mvn clean compile exec:java -Phello-world
```

[Read more about it](src/main/java/com/deepnetts/examples/helloworld/README.md).

### 2. Credit-card fraud

Flag suspicious card transactions before they cost money, without blocking good customers.

**You learn:** The whole workflow, explained in the most detail: read a CSV, train a network, test it against a baseline, save it, load it and predict. A yes/no network answers with a probability; a threshold turns it into yes or no.

**Takes:** about 10 seconds.

```bash
mvn clean compile exec:java -Pcredit-card-fraud
```

[Read more about it](src/main/java/com/deepnetts/examples/classification/creditcardfraud/README.md).

### 3. Taxi fare (linear regression)

Price a taxi ride from its distance and duration, and see the formula behind the price.

**You learn:** Linear regression: a network without a hidden layer is a formula (start price + price per km + price per minute), and you can read that formula out of its weights.

**Takes:** about 10 seconds.

```bash
mvn clean compile exec:java -Ptaxi-fare
```

[Read more about it](src/main/java/com/deepnetts/examples/regression/taxifare/README.md).

### 4. Exam pass (logistic regression)

Tell which students will pass an exam, and see how much each kind of preparation helps.

**You learn:** Logistic regression, the simplest yes/no model: a network without a hidden layer and with a SIGMOID output. Its weights show how much each input pushes toward "yes".

**Takes:** about 10 seconds.

```bash
mvn clean compile exec:java -Pexam-pass
```

[Read more about it](src/main/java/com/deepnetts/examples/classification/exampass/README.md).

### 5. Light switches (XOR)

A staircase light with a switch at the bottom and one at the top: why some rules need a hidden layer.

**You learn:** Why hidden layers exist: a network without one can only draw a straight line between "yes" and "no", and the XOR rule can't be split by a straight line. A hidden layer can learn it.

**Takes:** about 10 seconds.

```bash
mvn clean compile exec:java -Plight-switches
```

[Read more about it](src/main/java/com/deepnetts/examples/classification/lightswitch/README.md).

### 6. Spam email

Is an email spam?

**You learn:** Counts with a long tail (a few emails have very many `!`) go on a log scale, so the usual values are not squeezed together.

**Takes:** about 10 seconds.

```bash
mvn clean compile exec:java -Pspam-email
```

[Read more about it](src/main/java/com/deepnetts/examples/classification/spamemail/README.md).

### 7. Sonar mine

Is an object on the seabed a mine?

**You learn:** A category input (the seabed type) becomes several 0/1 inputs, one per category (one-hot), so the network doesn't read an order into it.

**Takes:** about 10 seconds.

```bash
mvn clean compile exec:java -Psonar-mines
```

[Read more about it](src/main/java/com/deepnetts/examples/classification/sonarmines/README.md).

### 8. Iris flower

Which of three iris species is a flower?

**You learn:** More than two answers: the target is one-hot (a 1 at the right species), the output layer uses SOFTMAX so the three probabilities add up to 1, and the test results are a 3 x 3 table instead of yes/no counts.

**Takes:** about 10 seconds.

```bash
mvn clean compile exec:java -Piris-flower
```

[Read more about it](src/main/java/com/deepnetts/examples/classification/irisflower/README.md).

### 9. Home value

What is the typical home value in a neighborhood?

**You learn:** Many inputs of different kinds (rates, counts, distances) in one regression.

**Takes:** about 10 seconds.

```bash
mvn clean compile exec:java -Pboston-housing
```

[Read more about it](src/main/java/com/deepnetts/examples/regression/bostonhousing/README.md).

### 10. CPU performance

How fast is a CPU?

**You learn:** Very uneven values (memory from 64 KB to 64 MB) go on a log scale, in the inputs and in the answer.

**Takes:** about 10 seconds.

```bash
mvn clean compile exec:java -Pcpu-performance
```

[Read more about it](src/main/java/com/deepnetts/examples/regression/cpuperformance/README.md).

### 11. Auto insurance

What will a policy group cost per insured year?

**You learn:** Some of the answer is chance: a small group has few claims, so its cost is partly luck, and no network can predict luck.

**Takes:** about 20 seconds.

```bash
mvn clean compile exec:java -Pauto-insurance
```

[Read more about it](src/main/java/com/deepnetts/examples/regression/autoinsurance/README.md).

### 12. Sorting algorithm

Which of 9 sorting algorithms is the fastest for an array?

**You learn:** Predict a number for every option, then choose: the network predicts each algorithm's time and the fastest is picked. The times were measured on a real computer, and they go from 0.1 microseconds to 38 ms, so the network learns their log.

**Takes:** about 20 seconds.

```bash
mvn clean compile exec:java -Psorting-algorithm
```

[Read more about it](src/main/java/com/deepnetts/examples/regression/sortingalgorithm/README.md).

### 13. Handwritten digits

Which digit, 0 to 9, is in a small gray image?

**You learn:** Ten answers from images: gray images (one value per pixel instead of three), SOFTMAX over 10 outputs and a 10 x 10 table of mistakes.

**Takes:** about 40 seconds.

```bash
mvn clean compile exec:java -Pdigits
```

[Read more about it](src/main/java/com/deepnetts/examples/image_classification/digits/README.md).

### 14. Duke detection

Does an image show Duke, the Java mascot?

**You learn:** The first convolutional network: filters that slide over the image, max pooling, then the usual layers. Images instead of a CSV.

**Takes:** about 20 seconds.

```bash
mvn clean compile exec:java -Pduke-detection
```

[Read more about it](src/main/java/com/deepnetts/examples/image_classification/dukedetection/README.md).

### 15. Lego figure

Does a picture show a Lego figure?

**You learn:** Choosing the other class: the "not Lego" pictures include Duke drawings, which have the same white background, so the network can't decide by the background alone.

**Takes:** about 3 minutes.

```bash
mvn clean compile exec:java -Plego-figure
```

[Read more about it](src/main/java/com/deepnetts/examples/image_classification/lego/README.md).

### 16. Parking spot

Is a parking spot taken?

**You learn:** Split by time, not at random: the camera photographs the lot every 5 minutes, so a random split would test on almost the same pictures it trained on. Here the test set is the end of the day.

**Takes:** about 40 seconds.

```bash
mvn clean compile exec:java -Pparking-spot
```

[Read more about it](src/main/java/com/deepnetts/examples/image_classification/parking/README.md).

### 17. Hot dog

Does a photo show a hot dog?

**You learn:** The limit of a small network: real food photos are all different (light, angle, background), and 300 training photos are too few to learn that from scratch. And what really helps instead: a pretrained network (next example).

**Takes:** about 3 minutes.

```bash
mvn clean compile exec:java -Phot-dog
```

[Read more about it](src/main/java/com/deepnetts/examples/image_classification/hotdog/README.md).

### 18. VGG16 import

What does an image show, out of 1000 everyday things?

**You learn:** Using a network trained somewhere else: VGG16 was trained in TensorFlow on 1.2 million ImageNet photos. Deep Netts builds the same layers, copies the weights in and runs it in pure Java, without training.

**Takes:** about 5 minutes.

This one needs about 4 GB of free memory and downloads 600 MB the first time. From the folder with `pom.xml`, run the command for your system, which also gives Maven the memory:

On Windows PowerShell:

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

[Read more about it](src/main/java/com/deepnetts/examples/image_classification/vgg16/README.md).

## Run it from an IDE

You can also run the examples without the command line, in any IDE that opens Maven projects. In IntelliJ IDEA:

1. **File → Open**, choose the `pom.xml` in this folder and **Open as Project**. Wait until the Maven import has finished.
2. **File → Project Structure → Project**: set the SDK to your JDK 25.
3. Open the main class of an example (for example `HelloWorld.java`) and click the green arrow next to `public static void main`.

The working directory must be this folder, so that the programs find `datasets/`. This is the default when you open `pom.xml` as a project.

For the VGG16 example (18) also add `-Xmx4g` in the run configuration under **Modify options → Add VM options**. Without it Java runs out of memory, and the program tells you so.

## If something goes wrong

| You see | What to do |
|---|---|
| `These examples need Java 25 or newer, but Maven is running on Java 17...` | Install a JDK 25 and point `JAVA_HOME` to it. `mvn --version` shows which Java Maven uses. |
| `mvn` is not recognized / command not found | Install Maven and add its `bin` folder to your `PATH`. |
| `there is no POM in this directory` | You are in the wrong folder, for example the folder of an example. Go to the folder with `pom.xml` (the root of the repository) and run the command again. |
| `File not found: datasets/...` | The program must run from the folder with `pom.xml`. In an IDE, set the working directory to it. |
| `The requested profile "..." could not be activated because it does not exist` and Hello World runs instead | The name after `-P` has a typo: Maven only warns and runs the default example. Check it against the table above. |
| `VGG16 needs about 4 GB of memory...` or `OutOfMemoryError` | Use the command for your system in example 18: it sets the memory. Close other programs first. |
| IntelliJ: `Module not specified` | The Maven project is not imported. Do step 1 of the IDE section again. |
| IntelliJ: `Project has no JDK configured` | Do step 2 of the IDE section. |

## Test

```bash
mvn test
```

This takes a few seconds: it checks the CSV reading, the sorting algorithms and that every input is between 0 and 1.
To also run every example twice and check that the output is clean and always the same (about 10 minutes):

```bash
mvn test -Pall
```

## How the code is organised

```
pom.xml                          one Maven project for all examples
datasets/                        the data of every example (CSV files and image folders)
src/main/java/com/deepnetts/examples/
    <group>/<example>/           one example
        <Example>.java           the whole workflow, from top to bottom: start reading here
        Your<Input>.java         the input you change
        Report.java              only prints the results
        dto/                     the input and the output as small records
        pipeline/                turns an input into the numbers the network uses, and checks it
        README.md                what the example shows, its output and its files
    util/                        small interfaces shared by the examples
```

## Learn more and get Deep Netts

- **[Getting started: the Deep Netts cookbook](https://www.deepnetts.com/cookbook/getting-started/)**: a code-first course that teaches neural networks with Java.
- **[Download and compare the editions](https://www.deepnetts.com/download/)**: the Get Started page with the Community Edition, the Free Edition and the other options.
- **[Community Edition on GitHub](https://github.com/deepnetts/deepnetts-communityedition)**: the open source code.

### Open source and the free option

You can start for free with either of two editions:

| | Community Edition | Free Edition |
|---|---|---|
| What it is | The open source core of Deep Netts: a pure Java deep learning engine, developed in the open on GitHub | The full accelerated platform |
| Price | Free | $0 |
| CPU acceleration with the Vector API | No | Yes |
| GPU acceleration with CUDA | No | Yes |
| Visual AI Builder | No | Yes |
| Free for | Learning, research and building | Development, testing and evaluation, and one production deployment while your product is small |
| Get it | [Maven Central](https://central.sonatype.com/artifact/com.deepnetts/deepnetts-core) or [GitHub](https://github.com/deepnetts/deepnetts-communityedition) | Free download: create an account on the [download page](https://www.deepnetts.com/download/) |

- **Vector API** is Java's way to use the vector instructions of the processor. The Free Edition uses it to run the calculations faster on the CPU.
- **GPU acceleration with CUDA** runs the calculations on an NVIDIA graphics card. It is part of the Free Edition, not of the Community Edition.

The Free Edition is meant for small products: one production environment, an annual revenue from the product under $100,000, and not offered as a hosted AI platform, managed service or SaaS.
Development, testing and evaluation are always free. The terms are on the [download page](https://www.deepnetts.com/download/). For larger needs there are the Enterprise Edition and a Managed Service.

The examples in this repository use only the Community Edition (`deepnetts-core` 4.0.5 from Maven Central) and run on the CPU. To use it in your own Maven project, add:

```xml
<dependency>
    <groupId>com.deepnetts</groupId>
    <artifactId>deepnetts-core</artifactId>
    <version>4.0.5</version>
</dependency>
```
