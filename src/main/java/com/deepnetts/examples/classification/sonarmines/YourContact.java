package com.deepnetts.examples.classification.sonarmines;

import com.deepnetts.examples.classification.sonarmines.dto.SonarContact;
import com.deepnetts.examples.classification.sonarmines.enums.SeabedType;

// This is the sonar contact the network judges.
// You can create your own contact and see what the network thinks:
// 1. Change the values (or try the example at the bottom).
// 2. Run SonarMines again.
// 3. Look at the output.
public final class YourContact {

    private YourContact() {
    }

    public static SonarContact create() {
        // Suspicious: close, strong echo, high-pitched, short shadow, on sand.
        return new SonarContact(
                35.0,                // range in meters
                90,                  // aspect angle in degrees, 0-180
                -12.0,               // echo level in dB (closer to 0 = stronger)
                0.9,                 // echo duration in ms
                105.0,               // spectral centroid in kHz
                4.0,                 // shadow length in meters
                SeabedType.SAND);    // seabed type (see SeabedType for all options)

        // Let's try the opposite: far away, weak and long echo, low-pitched, on rock.
        // To try it, comment out the suspicious one above and uncomment the line below.
        //return new SonarContact(70.0, 120, -30.0, 1.8, 80.0, 9.0, SeabedType.ROCK);
    }
}
