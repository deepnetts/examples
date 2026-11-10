package com.deepnetts.examples.regression.autoinsurance;

import com.deepnetts.examples.regression.autoinsurance.dto.PolicyGroup;
import com.deepnetts.examples.regression.autoinsurance.enums.Zone;

// This is the policy group the network judges.
// You can create your own policy group and see what the network thinks:
// 1. Change the values (or try the example at the bottom).
// 2. Run AutoInsurance again.
// 3. Look at the output.
public final class YourPolicyGroup {

    private YourPolicyGroup() {
    }

    public static PolicyGroup create() {
        // Risky: a big city, lots of driving, no bonus (a new driver), and a make with high claims.
        return new PolicyGroup(
                5,              // kilometres class, 1-5
                Zone.METRO,     // zone (see Zone for all options)
                1,              // bonus class, 1-7 (1 = no bonus)
                4,              // vehicle make, 1-9
                500.0);

        // Let's try the opposite: a quiet northern area, little driving, a long claim-free history.
        // To try it, comment out the one above and uncomment the line below.
        //return new PolicyGroup(1, Zone.RURAL_NORTH, 7, 8, 500.0);
    }
}
