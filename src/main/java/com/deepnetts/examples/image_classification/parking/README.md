# 16. Parking spot

**Is a parking spot taken?**

**What you learn here:** Split by time, not at random: the camera photographs the lot every 5 minutes, so a random split would test on almost the same pictures it trained on. Here the test set is the end of the day.

## Run it

From the root folder of the repository (the one with `pom.xml`, not this example's folder):

```bash
mvn clean compile exec:java -Pparking-spot
```

It takes about 40 seconds.

Then change the input in [YourImage.java](YourImage.java) (the path of an image file) and run it again.

## Result

93.0% correct on 1275 test pictures from the end of the day (baseline 57.6%). On 4 seeds 92-98%. Most mistakes are taken spots in the evening light, called free.

Results are on test data the network never saw, with seed 42 (the one in the code); **on 4 seeds** shows the range with seeds 42, 1, 2 and 3. **Baseline** is the result without a network: always the most common answer, or always the average.

## Limitations

- One camera, one parking lot, one sunny day. Another lot, angle or weather needs new pictures and new training.

## Data

[`datasets/ParkingSet`](../../../../../../../../datasets/ParkingSet): pictures of parking spots taken by one camera on one day, every 5 minutes (the file name is the date, the time and the spot).

## Files

Start reading in `ParkingSpot.java`: the whole workflow, from top to bottom.

| File | What it does |
|---|---|
| [`ParkingSpot.java`](ParkingSpot.java) | Parking spot example: a camera looks down at a parking lot and takes a picture every 5 minutes. |
| [`Report.java`](Report.java) | Prints the output, so ParkingSpot contains only the workflow. |
| [`YourImage.java`](YourImage.java) | This is the picture of a parking spot the network judges. |
| [`dto/ParkingData.java`](dto/ParkingData.java) | The parking images, split by the time of day they were taken. |
| [`dto/SpotPrediction.java`](dto/SpotPrediction.java) | The output of the network: how sure it is that a car is parked in the spot. |
| [`pipeline/SpotEncoder.java`](pipeline/SpotEncoder.java) | Bridge between images and the network, and between the image folder and the network. |
| [`pipeline/SpotFileReader.java`](pipeline/SpotFileReader.java) | Reads one image from a file and checks that it is really an image. |
| [`pipeline/SpotPredictor.java`](pipeline/SpotPredictor.java) | Predicts if a parking spot is taken. |

The interfaces `Predictor`, `ToRecord`, `ToTensor` and `CsvEncoder` are in [`util/`](../../util), shared by all examples.

---

Previous: [15. Lego figure](../lego) · [All examples](../../../../../../../../README.md#the-examples-in-a-suggested-order) · Next: [17. Hot dog](../hotdog)
