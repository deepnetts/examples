package com.deepnetts.examples.regression;

import com.deepnetts.examples.regression.autoinsurance.pipeline.PolicyEncoder;
import com.deepnetts.examples.regression.bostonhousing.pipeline.NeighborhoodEncoder;
import com.deepnetts.examples.regression.cpuperformance.pipeline.CpuEncoder;
import com.deepnetts.examples.regression.sortingalgorithm.pipeline.RunEncoder;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

// The regression examples teach the network a value on a small scale and turn its answer back into real units.
// Going there and back must give the same number, and the scaled values must stay between 0 and 1.
class TargetScalingTest {

    @Test
    void cpuPerformanceGoesThereAndBack() {
        assertEquals(500.0, CpuEncoder.unscaleTarget(CpuEncoder.scaleTarget(500)), 0.01);
        assertTrue(CpuEncoder.scaleTarget(1150) < 1 && CpuEncoder.scaleTarget(1) > 0);
    }

    @Test
    void homeValueGoesThereAndBack() {
        assertEquals(22.5, NeighborhoodEncoder.unscaleTarget(NeighborhoodEncoder.scaleTarget(22.5)), 0.001);
        assertTrue(NeighborhoodEncoder.scaleTarget(50) <= 1);
    }

    @Test
    void autoInsuranceGoesThereAndBack() {
        assertEquals(5.0, PolicyEncoder.unscaleTarget(PolicyEncoder.scaleTarget(5.0)), 0.001);
        assertTrue(PolicyEncoder.scaleTarget(31) < 1);
    }

    @Test
    void sortingTimeIsOnALogScaleAndGoesThereAndBack() {
        assertEquals(1000.0, RunEncoder.unscaleTarget(RunEncoder.scaleTarget(1000)), 0.01);
        assertEquals(0.1, RunEncoder.unscaleTarget(RunEncoder.scaleTarget(0.1)), 0.0001);
        // ten times longer is always the same step on the scale, for a fast and for a slow run
        float step1 = RunEncoder.scaleTarget(100) - RunEncoder.scaleTarget(10);
        float step2 = RunEncoder.scaleTarget(10_000) - RunEncoder.scaleTarget(1_000);
        assertEquals(step1, step2, 1e-6);
    }
}
