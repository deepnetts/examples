package com.deepnetts.examples.regression.cpuperformance.pipeline;

import com.deepnetts.examples.regression.cpuperformance.dto.Cpu;

import java.util.List;

// Turns a CPU into the numbers the network uses.
public final class FeatureExtractor {

    // Name and order of every input. The order must match extract().
    public static final List<String> FEATURE_NAMES = List.of(
            "log_machine_cycle_ns", "log_min_memory_kb", "log_max_memory_kb",
            "log_cache_memory_kb", "log_min_channels", "log_max_channels");

    public static final int NUM_FEATURES = FEATURE_NAMES.size();

    // The largest values in the data. Dividing by them puts every input roughly between 0 and 1,
    // so no input dominates.
    private static final double MAX_MACHINE_CYCLE_NS = 1500;
    private static final double MAX_MEMORY_MIN_KB    = 32_000;
    private static final double MAX_MEMORY_MAX_KB    = 64_000;
    private static final double MAX_CACHE_KB         = 256;
    private static final double MAX_CHANNELS_MIN     = 52;
    private static final double MAX_CHANNELS_MAX     = 176;

    private FeatureExtractor() {
    }

    public static float[] extract(Cpu cpu) {
        return new float[] {
                logScaled(cpu.machineCycleNs(), MAX_MACHINE_CYCLE_NS),
                logScaled(cpu.minMemoryKb(), MAX_MEMORY_MIN_KB),
                logScaled(cpu.maxMemoryKb(), MAX_MEMORY_MAX_KB),
                logScaled(cpu.cacheMemoryKb(), MAX_CACHE_KB),
                logScaled(cpu.minChannels(), MAX_CHANNELS_MIN),
                logScaled(cpu.maxChannels(), MAX_CHANNELS_MAX)
        };
    }

    // log1p(x) is log(1 + x). Memory goes from a few hundred KB to tens of thousands, and most CPUs
    // sit at the low end: using log so the very large values don't dominate. It also works for 0.
    // Dividing by log1p(max) then puts the result between 0 and 1.
    private static float logScaled(double value, double max) {
        return (float) (Math.log1p(value) / Math.log1p(max));
    }
}
