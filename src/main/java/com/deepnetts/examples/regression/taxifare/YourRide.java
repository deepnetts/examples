package com.deepnetts.examples.regression.taxifare;

import com.deepnetts.examples.regression.taxifare.dto.Ride;

// This is the ride the network prices.
// You can create your own ride and see what the network thinks:
// 1. Change the values (or try the example at the bottom).
// 2. Run TaxiFare again.
// 3. Look at the output.
public final class YourRide {

    private YourRide() {
    }

    public static Ride create() {
        // A ride across town: 12 km in 30 minutes.
        return new Ride(
                12.0,    // distance in km
                30.0);   // duration in minutes

        // Let's try a longer one: to the airport, 28 km in 55 minutes.
        // To try it, comment out the ride above and uncomment the line below.
        //return new Ride(28.0, 55.0);
    }
}
