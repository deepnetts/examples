# 4. Exam pass (logistic regression)

**Tell which students will pass an exam, and see how much each kind of preparation helps.**
A network without a hidden layer learns from 1,500 students, then gives a new student a probability to pass.

| | |
|---|---|
| You learn | Logistic regression, the simplest yes/no model: a network without a hidden layer and with a SIGMOID output. Its weights show how much each input pushes toward "yes". |
| You need | JDK 25 or newer and Maven 3.9 or newer. Deep Netts (Community Edition) is downloaded by Maven the first time. |
| It takes | About 6 seconds. The first run takes longer: Maven downloads Deep Netts (about 20 MB). |

## Run it

From the root folder of the repository (the one with `pom.xml`, not this example's folder):

```bash
mvn clean compile exec:java -Pexam-pass
```

In an IDE, open `pom.xml` as a project and run `ExamPass` (the working directory is the project folder).

## What you'll see

The same on every run:

```
--- 1. Dataset -----------------------------------------------------------------
  Total:      1500 students
  Passed:      873 (58.2%)
  Failed:      627 (41.8%)
  Inputs:        2 numbers per student (study hours, practice tests)

--- 2. Training ----------------------------------------------------------------
  Training the network...
  Stopped after 630 epochs

--- 3. Evaluation --------------------------------------------------------------

  Results on 299 students the network was never trained on:
  Threshold: 0.50 - a probability at or above it counts as PASS

                   Predicted FAIL   Predicted PASS
  Actual FAIL            105 (TN)          29 (FP)
  Actual PASS             27 (FN)         138 (TP)
  TN and TP are right answers, FP and FN are wrong ones.

  Accuracy  81.3%  - 243 of 299 correct
  Baseline  55.2%  - always answering with the more common answer would be this accurate

--- 4. What the network learned ------------------------------------------------

  A network without a hidden layer adds up one weight per input. Read out of it:

                          start      per hour        per test
  Learned:                -6.32         0.147           0.666
  The data's rule:        -6.00         0.150           0.600

  So one practice test counts as much as 4.5 hours of study (the data's rule: 4.0).

--- 5. Your student ------------------------------------------------------------
  Study hours:     20.0
  Practice tests:  3

  Pass probability:  20.0%
  Prediction:        FAIL

  Result: The network expects your student to fail.

  Edit YourStudent.java and run again to explore how the result changes.
```

## What happened

1. **Data:** every CSV row became a `Student` and its label, the correct answer (passed or failed). 20% of the students were kept aside for the test.
2. **Training:** the network adds up `start + weight1 * hours + weight2 * tests` and SIGMOID turns the sum into a probability from 0 to 1. Training found the weights that fit the students best.
3. **Evaluation:** on students it never saw, 81.3% of the answers are right. Always answering "pass" is right 55.2% of the time.
4. **What it learned:** each study hour adds 0.147 to the sum and each practice test 0.666, close to the rule the data was made with (0.150 and 0.600). So one practice test is worth about as much as 4 to 5 hours of study.
5. **Save, load, predict:** the network was saved to `models/exam-pass.dnet`, loaded back and asked about `YourStudent`.

## Try this

- In [`YourStudent.java`](YourStudent.java), use the student at the bottom (8 hours, 7 practice tests: 4 more tests, 12 hours less) and run again: the pass probability rises from 20.0% to 38.2%, because 4 tests count for more than 12 hours.
- Change `SEED` to `1`, `2` or `3`: 81-86% of the test students are classified correctly, and one practice test always counts as 4.2-4.5 hours of study.

## Limitations

- The data is synthetic: whether a student passed was decided by the rule above and then by chance. Near the 50% line, chance decides, so no model can be right much more often than about 85% here.
- A network without a hidden layer can only draw one straight line between "pass" and "fail". The [Light switches](../lightswitch) example shows a rule it can't learn.

## Data

[`datasets/exam-pass.csv`](../../../../../../../../datasets/exam-pass.csv): 1,500 synthetic students, made for this example.
The chance to pass follows `-6.0 + 0.15 * studyHours + 0.60 * practiceTests` (turned into a probability by a sigmoid), then chance decides.

## Files

Start reading in `ExamPass.java`: the whole workflow, from top to bottom.

| File | What it does |
|---|---|
| [`ExamPass.java`](ExamPass.java) | The steps: data, training, evaluation, what the network learned, save, load and predict. Start here. |
| [`YourStudent.java`](YourStudent.java) | The student the network judges. Change it and run again. |
| [`Report.java`](Report.java) | Only prints the results. You can skip it while learning. |
| [`dto/Student.java`](dto/Student.java) | The input of the network: one student. |
| [`dto/PassPrediction.java`](dto/PassPrediction.java) | The output of the network: the probability to pass and the final answer. |
| [`dto/PassWeights.java`](dto/PassWeights.java) | How much each input pushes toward passing. |
| [`pipeline/StudentEncoder.java`](pipeline/StudentEncoder.java) | Turns CSV rows and students into the numbers the network learns from, and reads the learned weights out of the network. |
| [`pipeline/FeatureExtractor.java`](pipeline/FeatureExtractor.java) | Puts the study hours and the practice tests between 0 and 1. |
| [`pipeline/StudentValidator.java`](pipeline/StudentValidator.java) | Checks that a student has possible values, with a clear error if not. |
| [`pipeline/PassPredictor.java`](pipeline/PassPredictor.java) | Student in, PassPrediction out: the class your application would use. |

The interfaces `Predictor`, `ToRecord`, `ToTensor` and `CsvEncoder` are in [`util/`](../../util), shared by all examples.

## Words you'll meet

| Word | Meaning |
|---|---|
| logistic regression | the simplest yes/no model: a network without a hidden layer and with a SIGMOID output |
| weight | how much one input pushes the answer toward "yes" (negative: toward "no") |
| sigmoid | squeezes any number into 0 to 1, so it can be read as a probability |
| label | the correct answer the network learns: here `passed` |
| threshold | the probability from which the answer is PASS |
| baseline | the result without a network: here, always answering "pass" |
| tensor | Deep Netts' array of numbers: what the network gets as input and gives as output |

---

Previous: [3. Taxi fare (linear regression)](../../regression/taxifare) · [All examples](../../../../../../../../README.md#the-examples-in-a-suggested-order) · Next: [5. Light switches (XOR)](../lightswitch)
