package com.deepnetts.examples.regression.cpuperformance.dto;

// The input of the network: one CPU (computer processor) and its memory setup.
public record Cpu(
        int machineCycleNs,    // time of one machine cycle, in nanoseconds (shorter = faster CPU)
        int minMemoryKb,       // minimum main memory, in kilobytes
        int maxMemoryKb,       // maximum main memory, in kilobytes
        int cacheMemoryKb,     // cache memory (small, very fast memory), in kilobytes
        int minChannels,       // minimum number of channels (connections to disks and other devices)
        int maxChannels) {     // maximum number of channels
}
