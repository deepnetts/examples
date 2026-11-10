# 2. Credit-card fraud

**Flag suspicious card transactions before they cost money, without blocking good customers.**
A neural network learns from 1,200 past transactions and gives every new one a fraud score.

| | |
|---|---|
| You learn | The whole workflow, explained in the most detail: read a CSV, train a network, test it against a baseline, save it, load it and predict. A yes/no network answers with a probability; a threshold turns it into yes or no. |
| You need | JDK 25 or newer and Maven 3.9 or newer. Deep Netts (Community Edition) is downloaded by Maven the first time. |
| It takes | About 6 seconds. The first run takes longer: Maven downloads Deep Netts (about 20 MB). |

## Run it

From the root folder of the repository (the one with `pom.xml`, not this example's folder):

```bash
mvn clean compile exec:java -Pcredit-card-fraud
```

In an IDE, open `pom.xml` as a project and run `CreditCardFraud` (the working directory is the project folder).

## What you'll see

The same on every run:

```
--- 1. Dataset -----------------------------------------------------------------
  Total:      1200 transactions
  Legitimate:  700 (58.3%)
  Fraud:       500 (41.7%)
  Inputs:       19 numbers per transaction (the features the network learns from)

--- 2. Training ----------------------------------------------------------------
  Training the network...
  Stopped after 70 epochs

--- 3. Evaluation --------------------------------------------------------------

  Results on 239 transactions the network was never trained on:
  Threshold: 0.50 - a probability at or above it counts as FRAUD

  Caught correctly:   78 fraud  + 105 legitimate = 183 correct  (76.6%)
  Got wrong:          24 fraud missed + 32 false alarms = 56 errors (23.4%)

  Of the 56 errors:
    24 missed fraud   - real fraud the network called legitimate
    32 false alarms   - legitimate transactions wrongly flagged

                     Predicted LEGITIMATE   Predicted FRAUD
  Actual LEGITIMATE              105 (TN)           32 (FP)
  Actual FRAUD                    24 (FN)           78 (TP)
  TN and TP are right answers, FP is a false alarm, FN is a missed fraud.

  Accuracy  76.6%  - 183 of 239 correct
  Baseline  57.3%  - always guessing the more common answer would be this accurate
  Precision 70.9%  - when flagged as FRAUD, it really is fraud (78 of 110 flagged)
  Recall    76.5%  - 78 of the 102 actual fraud cases were caught
  F1        0.736  - combined precision+recall score (1.0 = perfect)

--- 4. Your transaction --------------------------------------------------------
  Amount:             420.00
  Category:           shopping_net
  Time:               03:00
  Distance from home: 180.0 km

  Fraud probability:  99.7%
  Prediction:         FRAUD

  Result: The network flagged your transaction as fraud.

  Edit YourTransaction.java and run again to explore how the result changes.
```

## What happened

1. **Data:** every CSV row became a `Transaction` and its label, the correct answer (fraud or legitimate). 20% of the rows were kept aside for the test.
2. **Training:** the network looked at the training rows again and again (each pass is an *epoch*) and stopped when it stopped improving.
3. **Evaluation:** on transactions it never saw. Always answering "legitimate" is already right 57.3% of the time: the network has to beat that, and it does (76.6%).
4. **Save, load, predict:** the network was saved to `models/credit-card-fraud.dnet`, loaded back and asked about `YourTransaction`, the way your application would use it.

## Try this

- In [`YourTransaction.java`](YourTransaction.java), use the grocery purchase at the bottom (35.00, 2 PM, 4 km from home) and run again: the fraud probability drops to 15.3% and the answer becomes LEGITIMATE.
- In [`CreditCardFraud.java`](CreditCardFraud.java), set `THRESHOLD` to `0.3f`: more fraud is caught (recall goes from 76.5% to 88.2%), but there are more false alarms (precision goes from 70.9% to 62.5%).
- Change `SEED` to `1`, `2` or `3`: the results move by a few points. Over these 4 seeds, 75-80% of the test transactions are classified correctly.

## Limitations

- The data has far more fraud (42%) than real card data (well under 1%). So the "fraud probability" is a score: read it as "how suspicious", not as the real chance that a transaction is fraud.
- The test set has 239 transactions, so the results can move by a few points from one split to another (see `SEED` above).
- Deep Netts stops training a few checks after the best result and keeps the network from the last epoch, not the best one.

## Data

[`datasets/creditcard-fraud.csv`](../../../../../../../../datasets/creditcard-fraud.csv): 1,200 rows taken from the
[Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
by Kartik Shenoy (CC0: Public Domain). That data is synthetic: it was simulated with Brandon Harris's
[Sparkov Data Generation](https://github.com/namebrandon/Sparkov_Data_Generation) tool, so no real person or card is in it.
The hour of day and the distance from home were computed from the original fields.

## Files

Start reading in `CreditCardFraud.java`: the whole workflow, from top to bottom.

| File | What it does |
|---|---|
| [`CreditCardFraud.java`](CreditCardFraud.java) | The four steps: data, training, evaluation, save, load and predict. Start here. |
| [`YourTransaction.java`](YourTransaction.java) | The transaction the network judges. Change it and run again. |
| [`Report.java`](Report.java) | Only prints the results. You can skip it while learning. |
| [`dto/Transaction.java`](dto/Transaction.java) | The input of the network: one credit-card transaction. |
| [`dto/FraudPrediction.java`](dto/FraudPrediction.java) | The output of the network: the fraud probability and the final answer. |
| [`enums/MerchantCategory.java`](enums/MerchantCategory.java) | The merchant categories the network knows. |
| [`pipeline/TransactionEncoder.java`](pipeline/TransactionEncoder.java) | Turns CSV rows and transactions into the numbers the network learns from. |
| [`pipeline/FeatureExtractor.java`](pipeline/FeatureExtractor.java) | The math that turns a transaction into 19 numbers (features): logarithms, the hour as a circle, one 0/1 number per category. |
| [`pipeline/TransactionValidator.java`](pipeline/TransactionValidator.java) | Checks that a transaction has possible values, with a clear error if not. |
| [`pipeline/FraudPredictor.java`](pipeline/FraudPredictor.java) | Transaction in, FraudPrediction out: the class your application would use. |

The interfaces `Predictor`, `ToRecord`, `ToTensor` and `CsvEncoder` are in [`util/`](../../util), shared by all examples.

## Words you'll meet

| Word | Meaning |
|---|---|
| feature | one number the network learns from (19 per transaction here) |
| label | the correct answer the network learns: here `is_fraud` |
| training set, test set | rows to learn from; rows kept aside to check the result honestly |
| epoch | one pass over all the training rows |
| baseline | the result without a network: here, always answering "legitimate" |
| threshold | the probability from which the answer is FRAUD |
| precision, recall | how many alarms were real fraud; how many of the frauds were caught |
| tensor | Deep Netts' array of numbers: what the network gets as input and gives as output |

---

Previous: [1. Hello World](../../helloworld) · [All examples](../../../../../../../../README.md#the-examples-in-a-suggested-order) · Next: [3. Taxi fare (linear regression)](../../regression/taxifare)
