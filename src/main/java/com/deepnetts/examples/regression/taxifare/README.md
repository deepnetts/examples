# 3. Taxi fare (linear regression)

**Price a taxi ride from its distance and duration, and see the formula behind the price.**
A network without a hidden layer learns from 1,500 past rides, then tells you what a new ride will cost.

| | |
|---|---|
| You learn | Linear regression: a network without a hidden layer is a formula (start price + price per km + price per minute), and you can read that formula out of its weights. |
| You need | JDK 25 or newer and Maven 3.9 or newer. Deep Netts (Community Edition) is downloaded by Maven the first time. |
| It takes | About 6 seconds. The first run takes longer: Maven downloads Deep Netts (about 20 MB). |

## Run it

From the root folder of the repository (the one with `pom.xml`, not this example's folder):

```bash
mvn clean compile exec:java -Ptaxi-fare
```

In an IDE, open `pom.xml` as a project and run `TaxiFare` (the working directory is the project folder).

## What you'll see

The same on every run:

```
--- 1. Dataset -----------------------------------------------------------------
  Total:      1500 taxi rides
  Inputs:        2 numbers per ride (distance in km, duration in minutes)

--- 2. Training ----------------------------------------------------------------
  Training the network...
  Stopped after 30 epochs

--- 3. Evaluation --------------------------------------------------------------

  Results on 299 rides the network was never trained on:

  Average error   0.63 EUR  - on average the predicted fare is off by this much
  Baseline       13.63 EUR  - always answering with the average fare would be off by this much
  R2             0.997      - share of the differences between fares explained (1.0 = all)

--- 4. What the network learned ------------------------------------------------

  A network without a hidden layer is a formula. Read out of its weights:

  Learned:          fare = 2.88 + 1.20 per km + 0.30 per minute
  The data's rule:  fare = 2.50 + 1.20 per km + 0.30 per minute (plus a little noise)

--- 5. Your ride ---------------------------------------------------------------
  Distance:  12.0 km
  Duration:  30 minutes

  Result: The network expects a fare of about 26.32 EUR.

  Edit YourRide.java and run again to explore how the result changes.
```

## What happened

1. **Data:** every CSV row became a `Ride` and its fare, the correct answer. 20% of the rides were kept aside for the test.
2. **Training:** the network has no hidden layer: its answer is `bias + weight1 * distance + weight2 * duration`. Training found the two weights and the bias that fit the rides best.
3. **Evaluation:** on rides it never saw, the fare is off by 0.63 EUR on average, about as much as the noise in the data. Always answering with the average fare would be off by 13.63 EUR.
4. **What it learned:** the two weights and the bias, turned back into EUR, are the fare formula: 1.20 per km and 0.30 per minute, exactly the rule the data was made with.
5. **Save, load, predict:** the network was saved to `models/taxi-fare.dnet`, loaded back and asked about `YourRide`.

## Try this

- In [`YourRide.java`](YourRide.java), use the airport ride at the bottom (28 km, 55 minutes) and run again: the network expects about 53.04 EUR.
- Change `SEED` to `1`, `2` or `3`: the price per km (1.19-1.21) and per minute (0.30) hardly move, the start price moves more (2.27-2.88). No ride is 0 km long, so the start price is the hardest number to pin down.

## Limitations

- The data is synthetic: every fare was made with the formula above, plus a little noise. Real fares also depend on the time of day, tolls and waiting, so a real formula would not fit this well.
- A straight-line formula is all this network can learn. When the real rule is not a straight line, add a hidden layer (and lose the readable formula).

## Data

[`datasets/taxi-fares.csv`](../../../../../../../../datasets/taxi-fares.csv): 1,500 synthetic taxi rides, made for this example.
Each fare is `2.50 + 1.20 * km + 0.30 * minutes`, plus noise of about 0.80 EUR.

## Files

Start reading in `TaxiFare.java`: the whole workflow, from top to bottom.

| File | What it does |
|---|---|
| [`TaxiFare.java`](TaxiFare.java) | The steps: data, training, evaluation, what the network learned, save, load and predict. Start here. |
| [`YourRide.java`](YourRide.java) | The ride the network prices. Change it and run again. |
| [`Report.java`](Report.java) | Only prints the results. You can skip it while learning. |
| [`dto/Ride.java`](dto/Ride.java) | The input of the network: one taxi ride. |
| [`dto/FarePrediction.java`](dto/FarePrediction.java) | The output of the network: the expected fare. |
| [`dto/FareFormula.java`](dto/FareFormula.java) | A fare formula: start price, price per km, price per minute. |
| [`pipeline/RideEncoder.java`](pipeline/RideEncoder.java) | Turns CSV rows and rides into the numbers the network learns from, and reads the learned formula out of the network. |
| [`pipeline/FeatureExtractor.java`](pipeline/FeatureExtractor.java) | Puts the distance and the duration between 0 and 1. |
| [`pipeline/RideValidator.java`](pipeline/RideValidator.java) | Checks that a ride has possible values, with a clear error if not. |
| [`pipeline/FarePredictor.java`](pipeline/FarePredictor.java) | Ride in, FarePrediction out: the class your application would use. |

The interfaces `Predictor`, `ToRecord`, `ToTensor` and `CsvEncoder` are in [`util/`](../../util), shared by all examples.

## Words you'll meet

| Word | Meaning |
|---|---|
| linear regression | predicting a number with a straight-line formula: a network without a hidden layer |
| weight, bias | the numbers the network learns: one weight per input, plus one bias (here: price per km, price per minute, start price) |
| feature | one number the network learns from (2 per ride here) |
| average error (MAE) | how far the predicted fares are from the real ones, on average |
| R2 | how much of the differences between fares the network explains: 1.0 = all of them |
| baseline | the result without a network: here, always answering with the average fare |
| tensor | Deep Netts' array of numbers: what the network gets as input and gives as output |

---

Previous: [2. Credit-card fraud](../../classification/creditcardfraud) · [All examples](../../../../../../../../README.md#the-examples-in-a-suggested-order) · Next: [4. Exam pass (logistic regression)](../../classification/exampass)
