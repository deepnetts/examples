package com.deepnetts.examples.regression.autoinsurance.dto;

import com.deepnetts.examples.regression.autoinsurance.enums.Zone;

// The input of the network: one group of insurance policies that share the same traits.
public record PolicyGroup(
        int kilometresClass,    // yearly driving distance class, 1-5 (1 = least driving, 5 = most)
        Zone zone,              // where the vehicles are mostly driven
        int bonusClass,         // 1-7: higher = more years without claims = bigger discount
        int vehicleMake,        // vehicle make as a code, 1-9 (a label, not a size)
        double insuredYears) {  // total insured years of the whole group (more years = more exposure to claims)
}
