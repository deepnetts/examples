# 6. Spam email

**Is an email spam?**

**What you learn here:** Counts with a long tail (a few emails have very many `!`) go on a log scale, so the usual values are not squeezed together.

## Run it

From the root folder of the repository (the one with `pom.xml`, not this example's folder):

```bash
mvn clean compile exec:java -Pspam-email
```

Then change the input in [YourEmail.java](YourEmail.java) (share of the words free, money, remove and your; share of ! and $; capital letters) and run it again.

## Result

89.6% correct (baseline 56.7%), finds 81% of spam. On 4 seeds: 89-92% correct.

Results are on test data the network never saw, with seed 42 (the one in the code); **on 4 seeds** shows the range with seeds 42, 1, 2 and 3. **Baseline** is the result without a network: always the most common answer, or always the average.

## Limitations

- Only 8 of the 57 Spambase columns, and the emails are from the late 1990s: today's spam looks different.

## Data

[`datasets/spam-email.csv`](../../../../../../../../datasets/spam-email.csv): 8 of the 57 columns of [Spambase](https://archive.ics.uci.edu/dataset/94/spambase) (UCI Machine Learning Repository, CC BY 4.0), without duplicate and contradicting rows.

## Files

Start reading in `SpamEmail.java`: the whole workflow, from top to bottom.

| File | What it does |
|---|---|
| [`Report.java`](Report.java) | Prints the output, so SpamEmail contains only the workflow. |
| [`SpamEmail.java`](SpamEmail.java) | Spam detection example: a neural network learns from past emails. |
| [`YourEmail.java`](YourEmail.java) | This is the email the network judges. |
| [`dto/Email.java`](dto/Email.java) | The input of the network: one email, described by simple word and character statistics. |
| [`dto/SpamPrediction.java`](dto/SpamPrediction.java) | The output of the network: spam probability and the final prediction. |
| [`pipeline/EmailEncoder.java`](pipeline/EmailEncoder.java) | Bridge between our records and the network. |
| [`pipeline/EmailValidator.java`](pipeline/EmailValidator.java) | Checks that an email, and every row of the CSV file, has valid values. |
| [`pipeline/FeatureExtractor.java`](pipeline/FeatureExtractor.java) | Turns an email into the numbers the network uses. |
| [`pipeline/SpamPredictor.java`](pipeline/SpamPredictor.java) | Predicts whether one email is spam. |

The interfaces `Predictor`, `ToRecord`, `ToTensor` and `CsvEncoder` are in [`util/`](../../util), shared by all examples.

---

Previous: [5. Light switches (XOR)](../lightswitch) · [All examples](../../../../../../../../README.md#the-examples-in-a-suggested-order) · Next: [7. Sonar mine](../sonarmines)
