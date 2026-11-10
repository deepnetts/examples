package com.deepnetts.examples.regression.cpuperformance;

import com.deepnetts.examples.regression.cpuperformance.dto.Cpu;

// This is the CPU the network judges.
// You can create your own CPU and see what the network thinks:
// 1. Change the values (or try the example at the bottom).
// 2. Run CpuPerformance again.
// 3. Look at the output.
public final class YourCpu {

    private YourCpu() {
    }

    public static Cpu create() {
        // A strong machine: fast cycle, lots of memory, big cache, many channels.
        return new Cpu(
                30,       // machine cycle time in ns
                8000,     // minimum memory in KB
                32000,    // maximum memory in KB
                64,       // cache memory in KB
                8,        // minimum channels
                32);

        // Let's try the opposite: a weak machine with a slow cycle, little memory and no big cache.
        // To try it, comment out the one above and uncomment the line below.
        //return new Cpu(400, 512, 2000, 4, 1, 4);
    }
}
