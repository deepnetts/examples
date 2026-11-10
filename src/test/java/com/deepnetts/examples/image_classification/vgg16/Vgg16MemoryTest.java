package com.deepnetts.examples.image_classification.vgg16;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

// VGG16 needs about 4 GB. With less, the example says so clearly instead of failing with OutOfMemoryError in the middle.
class Vgg16MemoryTest {

    private static final long GB = 1024L * 1024 * 1024;

    @Test
    void tooLittleMemoryGivesAClearMessage() {
        IllegalStateException e = assertThrows(IllegalStateException.class, () -> Vgg16Import.requireMemory(2 * GB));

        assertTrue(e.getMessage().contains("needs about 4 GB"), e.getMessage());
        assertTrue(e.getMessage().contains("only 2048 MB"), e.getMessage());
        assertTrue(e.getMessage().contains("-Xmx4g"), e.getMessage());
    }

    @Test
    void enoughMemoryPasses() {
        assertDoesNotThrow(() -> Vgg16Import.requireMemory(4 * GB));
    }
}
