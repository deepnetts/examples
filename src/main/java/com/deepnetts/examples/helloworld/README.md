# 1. Hello World

**Check that Java and Deep Netts work, in under a second. Run it first.**
The smallest possible network, one input and one output, learns to turn degrees Celsius into Fahrenheit.

| | |
|---|---|
| You learn | That the setup works, and the smallest training loop: make data, build a network, train it, ask it. |
| You need | JDK 25 or newer and Maven 3.9 or newer. Deep Netts (Community Edition) is downloaded by Maven the first time. |
| It takes | Under a second. The first run takes longer: Maven downloads Deep Netts (about 20 MB). |

## Run it

From the root folder of the repository (the one with `pom.xml`, not this example's folder):

```bash
mvn clean compile exec:java -Phello-world
```

In an IDE, open `pom.xml` as a project and run `HelloWorld` (the working directory is the project folder).

## What you'll see

The same on every run:

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

If you see `SUCCESS`, everything is ready for the other examples. Your Java version may be newer than 25.

## What happened

1. **Data:** there is no CSV file. The 15 examples are made in the code, from -40 C to 100 C, with the formula F = 1.8 * C + 32. Every temperature is divided by 100, because a network learns best with small numbers.
2. **Training:** the network computes `weight * input + bias`. Training looks for the weight and the bias that give the right Fahrenheit for every example: 1.8 and 32 (on the small scale). It stops when the error is tiny.
3. **Predict:** the trained network converts 37 C, a temperature it never saw, and gets 98.6 F.

## Try this

- In [`HelloWorld.java`](HelloWorld.java), change `YOUR_CELSIUS` to `100` and run again: the network says 211.9 F (exact: 212.0 F).
- Change it to `1000`, far outside the examples: the network says 1830.1 F (exact: 1832.0 F). A straight line keeps working outside the data, but the small errors in the weight grow with the number.

## Files

| File | What it does |
|---|---|
| [`HelloWorld.java`](HelloWorld.java) | The whole example in one file: the data, the network, the training and the prediction. |

## Words you'll meet

| Word | Meaning |
|---|---|
| network | the model that learns: here one input, one output and the weight and the bias between them |
| epoch | one pass over all the training examples |
| weight, bias | the two numbers training adjusts: the output is weight * input + bias |
| tensor | Deep Netts' array of numbers: what the network gets as input and gives as output |

---

[All examples](../../../../../../../README.md#the-examples-in-a-suggested-order) · Next: [2. Credit-card fraud](../classification/creditcardfraud)
