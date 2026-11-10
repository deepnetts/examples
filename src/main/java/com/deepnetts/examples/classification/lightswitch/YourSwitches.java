package com.deepnetts.examples.classification.lightswitch;

import com.deepnetts.examples.classification.lightswitch.dto.Switches;

// These are the switches the network judges.
// You can set your own switches and see what the network thinks:
// 1. Change the values (or try the example at the bottom).
// 2. Run LightSwitches again.
// 3. Look at the output.
public final class YourSwitches {

    private YourSwitches() {
    }

    public static Switches create() {
        // The bottom switch is up, the top switch is down.
        return new Switches(
                true,    // bottom switch up
                false);  // top switch up

        // Let's try both switches up.
        // To try it, comment out the switches above and uncomment the line below.
        //return new Switches(true, true);
    }
}
