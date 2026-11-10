package com.deepnetts.examples.regression.bostonhousing;

import com.deepnetts.examples.regression.bostonhousing.dto.Neighborhood;

// This is the neighborhood the network judges.
// You can create your own neighborhood and see what the network thinks:
// 1. Change the values (or try the example at the bottom).
// 2. Run BostonHousing again.
// 3. Look at the output.
public final class YourNeighborhood {

    private YourNeighborhood() {
    }

    public static Neighborhood create() {
        // A pricey area: quiet, big homes, new buildings, good schools, clean air, few lower-status residents.
        return new Neighborhood(
                0.05,    // crime rate
                7.8,     // average rooms per home
                25.0,    // old buildings, %
                5.0,     // distance to employment, km
                3,       // highway access index, 1-24
                14.5,    // pupils per teacher
                0.45,    // air pollution (nitrogen oxides), 0-1
                4.0);

        // Let's try the opposite: more crime, small and old homes, crowded schools, polluted air.
        // To try it, comment out the one above and uncomment the line below.
        //return new Neighborhood(12.0, 5.3, 90.0, 2.0, 24, 20.2, 0.70, 25.0);
    }
}
