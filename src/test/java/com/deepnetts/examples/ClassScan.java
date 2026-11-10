package com.deepnetts.examples;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Stream;

// Finds every class of the examples (in target/classes), sorted by name, so a test can check all of them.
final class ClassScan {

    private ClassScan() {
    }

    static List<Class<?>> all() throws IOException, URISyntaxException {
        Path root = Path.of(ClassScan.class.getProtectionDomain().getCodeSource().getLocation().toURI())
                .resolveSibling("classes");   // target/classes next to target/test-classes
        try (Stream<Path> files = Files.walk(root)) {
            return files.filter(file -> file.toString().endsWith(".class") && !file.getFileName().toString().contains("$"))
                    .map(file -> root.relativize(file).toString().replace('/', '.').replace('\\', '.'))
                    .map(name -> name.substring(0, name.length() - ".class".length()))
                    .map(ClassScan::load)
                    .sorted((a, b) -> a.getName().compareTo(b.getName()))
                    .toList();
        }
    }

    private static Class<?> load(String name) {
        try {
            return Class.forName(name);
        } catch (ClassNotFoundException e) {
            throw new IllegalStateException(e);
        }
    }
}
