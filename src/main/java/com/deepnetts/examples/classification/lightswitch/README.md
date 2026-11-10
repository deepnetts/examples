# 5. Light switches (XOR)

**A staircase light with a switch at the bottom and one at the top: why some rules need a hidden layer.**
The light is on when exactly one switch is up. Two networks try to learn that: one without and one with a hidden layer.

| | |
|---|---|
| You learn | Why hidden layers exist: a network without one can only draw a straight line between "yes" and "no", and the XOR rule can't be split by a straight line. A hidden layer can learn it. |
| You need | JDK 25 or newer and Maven 3.9 or newer. Deep Netts (Community Edition) is downloaded by Maven the first time. |
| It takes | About 6 seconds. The first run takes longer: Maven downloads Deep Netts (about 20 MB). |

## Run it

From the root folder of the repository (the one with `pom.xml`, not this example's folder):

```bash
mvn clean compile exec:java -Plight-switches
```

In an IDE, open `pom.xml` as a project and run `LightSwitches` (the working directory is the project folder).

## What you'll see

The same on every run:

```
--- 1. Dataset -----------------------------------------------------------------
  Total:      4 positions of the two switches (all there are)
  Rule:       the light is on when exactly one switch is up (XOR)

--- 2. Training ----------------------------------------------------------------
  Training two networks on the same 4 rows...
  Without a hidden layer:  stopped after 5000 epochs
  With a hidden layer:     stopped after 151 epochs

--- 3. Evaluation --------------------------------------------------------------

  Bottom   Top      Light   Without a hidden layer   With a hidden layer
  down     down     off     ?    50.3%  unsure       off   0.2%  right
  down     up       on      ?    50.1%  unsure       on   99.0%  right
  up       down     on      ?    50.1%  unsure       on   98.6%  right
  up       up       off     ?    50.0%  unsure       off   1.3%  right

  Without a hidden layer: 0 of 4 correct, 4 unsure (close to 50%)
  With a hidden layer:    4 of 4 correct, 0 unsure (close to 50%)

  No straight line separates the two "on" positions from the two "off" ones,
  so a network without a hidden layer can't learn this rule. A hidden layer can.

--- 4. Your switches -----------------------------------------------------------
  Bottom switch:  up
  Top switch:     down

  On probability:  98.6%
  Prediction:      ON

  Result: The network says the light is on.

  Edit YourSwitches.java and run again to explore how the result changes.
```

## What happened

1. **Data:** the CSV has all 4 positions of the two switches. With only 4 possible inputs, both networks learn from all of them and are checked on all of them: the question is not "does it work on new data", but "can it learn the rule at all".
2. **Training:** both networks get the same data and the same settings. The only difference is the hidden layer of 4 units.
3. **Evaluation:** the network without a hidden layer answers about 50% for every position: it can't decide. The network with a hidden layer is right on all 4, and sure of it (below 2% or above 98%).
4. **Save, load, predict:** the network with the hidden layer was saved to `models/light-switches.dnet`, loaded back and asked about `YourSwitches`.

## Why a straight line is not enough

Put the 4 positions on a square: bottom switch left/right, top switch down/up. The two "on" positions are on one diagonal,
the two "off" positions on the other. One straight line can't put both "on" corners on one side and both "off" corners on
the other. A network without a hidden layer draws exactly one such line. Each unit of a hidden layer draws its own line,
and the output combines them, so together they can cut out the two "on" corners.

## Try this

- In [`YourSwitches.java`](YourSwitches.java), use both switches up (the line at the bottom) and run again: the on probability drops to 1.3% and the light is off.
- In [`LightSwitches.java`](LightSwitches.java), change the hidden layer from 4 units to 1 and run again: with only one line, it gets 3 of the 4 positions right, never all 4.
- Change `SEED` to `1`, `2` or `3`: the network without a hidden layer stays at about 50%, the one with a hidden layer is always right on all 4.

## Limitations

- 4 rows is the whole rule, not a sample of it, so there is no test set and nothing to generalize to. The example shows what each kind of network *can* learn, not how well it works on new data.

## Data

[`datasets/light-switches.csv`](../../../../../../../../datasets/light-switches.csv): the 4 positions of the two switches (1 = up) and the light (1 = on). Made for this example.

## Files

Start reading in `LightSwitches.java`: the whole workflow, from top to bottom.

| File | What it does |
|---|---|
| [`LightSwitches.java`](LightSwitches.java) | The steps: data, training of both networks, evaluation, save, load and predict. Start here. |
| [`YourSwitches.java`](YourSwitches.java) | The switches the network judges. Change them and run again. |
| [`Report.java`](Report.java) | Only prints the results. You can skip it while learning. |
| [`dto/Switches.java`](dto/Switches.java) | The input of the network: the two switches. |
| [`dto/LightPrediction.java`](dto/LightPrediction.java) | The output of the network: the on probability and the final answer. |
| [`dto/TruthTableRow.java`](dto/TruthTableRow.java) | One position of the switches, the real light and what both networks say. |
| [`pipeline/SwitchesEncoder.java`](pipeline/SwitchesEncoder.java) | Turns CSV rows and switches into the numbers the network learns from (1 = up, 0 = down). |
| [`pipeline/LightPredictor.java`](pipeline/LightPredictor.java) | Switches in, LightPrediction out: the class your application would use. |

The interfaces `Predictor`, `ToRecord`, `ToTensor` and `CsvEncoder` are in [`util/`](../../util), shared by all examples.

## Words you'll meet

| Word | Meaning |
|---|---|
| XOR | "exclusive or": yes when exactly one of two things is true |
| hidden layer | units between the input and the output; each can learn its own pattern |
| unit | one small part of a layer: it adds up its inputs and squeezes the sum (here with TANH) |
| epoch | one pass over all the training rows |
| tensor | Deep Netts' array of numbers: what the network gets as input and gives as output |

---

Previous: [4. Exam pass (logistic regression)](../exampass) · [All examples](../../../../../../../../README.md#the-examples-in-a-suggested-order) · Next: [6. Spam email](../spamemail)
