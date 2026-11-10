package com.deepnetts.examples;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

// What a reader of the README does must work: every command names a profile that exists, every profile starts a class
// with a main method, every link leads somewhere, and the order of the examples and the links at the end are there.
class ProjectConsistencyTest {

    // Profiles that are not examples in the suggested order.
    private static final Set<String> NOT_IN_THE_ORDER = Set.of("all", "sorting-benchmark");

    private static final Pattern PROFILE = Pattern.compile(
            "<profile>\\s*<id>([\\w-]+)</id>\\s*<properties>\\s*<exec\\.mainClass>([^<]+)</exec\\.mainClass>");
    private static final Pattern COMMAND = Pattern.compile("(?m)^mvn clean compile exec:java -P([\\w-]+)$");
    private static final Pattern LINK = Pattern.compile("\\]\\(([^)\\s]+)\\)");

    private static String read(String file) throws IOException {
        return Files.readString(Path.of(file));
    }

    private static Map<String, String> profiles() throws IOException {
        Map<String, String> profiles = new LinkedHashMap<>();
        Matcher m = PROFILE.matcher(read("pom.xml"));
        while (m.find()) {
            profiles.put(m.group(1), m.group(2));
        }
        return profiles;
    }

    private static List<Path> readmes() throws IOException {
        try (Stream<Path> files = Files.walk(Path.of("."))) {
            return files.filter(p -> p.getFileName().toString().equals("README.md"))
                    .filter(p -> !p.toString().contains("target") && !p.toString().contains("models"))
                    .toList();
        }
    }

    @Test
    void everyCommandInTheReadmeNamesAnExistingProfile() throws IOException {
        Set<String> known = profiles().keySet();
        int commands = 0;
        for (Path readme : readmes()) {
            Matcher m = COMMAND.matcher(Files.readString(readme));
            while (m.find()) {
                commands++;
                assertTrue(known.contains(m.group(1)), readme + ": no profile named " + m.group(1) + " in pom.xml");
            }
        }
        assertTrue(commands >= 18, "found only " + commands + " commands");
    }

    @Test
    void everyProfileStartsAClassWithAMainMethod() throws Exception {
        for (Map.Entry<String, String> profile : profiles().entrySet()) {
            Class<?> type = Class.forName(profile.getValue());
            Method main = type.getMethod("main", String[].class);
            assertTrue(Modifier.isStatic(main.getModifiers()), profile.getKey() + ": main must be static");
        }
        Matcher defaultClass = Pattern.compile("<exec\\.mainClass>([^<]+)</exec\\.mainClass>\\s*<excludedTestGroups>").matcher(read("pom.xml"));
        assertTrue(defaultClass.find(), "pom.xml has no default exec.mainClass");
        assertNotNull(Class.forName(defaultClass.group(1)).getMethod("main", String[].class));
    }

    @Test
    void everyExampleIsInTheReadmeOnceInASuggestedOrder() throws IOException {
        String readme = read("README.md");
        assertTrue(readme.contains("## The examples in a suggested order"), "the root README has no suggested order");

        Set<String> examples = new TreeSet<>(profiles().keySet());
        examples.removeAll(NOT_IN_THE_ORDER);

        // The rows of the table: | 1 | [Title](...) | time | `-Pname` |
        Matcher row = Pattern.compile("(?m)^\\| (\\d+) \\| \\[[^\\]]+\\]\\([^)]+\\) \\| [^|]+ \\| `-P([\\w-]+)` \\|$").matcher(readme);
        List<String> inTable = new ArrayList<>();
        int expected = 1;
        while (row.find()) {
            assertEquals(expected++, Integer.parseInt(row.group(1)), "the numbers of the order must be 1, 2, 3...");
            inTable.add(row.group(2));
        }
        assertEquals(examples, new TreeSet<>(inTable), "the order must contain every example exactly once");
        assertEquals(inTable.size(), new TreeSet<>(inTable).size(), "an example is listed twice");
    }

    @Test
    void everyExampleReadmeLinksToThePreviousAndNextExample() throws IOException {
        int withFooter = 0;
        for (Path readme : readmes()) {
            if (readme.getParent().normalize().equals(Path.of("").normalize())) {
                continue;
            }
            String text = Files.readString(readme);
            if (text.contains("[All examples](")) {
                withFooter++;
            }
        }
        assertEquals(profiles().keySet().stream().filter(p -> !NOT_IN_THE_ORDER.contains(p)).count(), withFooter,
                "every example README needs its Previous / All examples / Next line");
    }

    @Test
    void everyLinkInEveryReadmeLeadsToAFile() throws IOException {
        int links = 0;
        for (Path readme : readmes()) {
            Matcher m = LINK.matcher(Files.readString(readme));
            while (m.find()) {
                String link = m.group(1);
                if (link.startsWith("http") || link.startsWith("#")) {
                    continue;
                }
                links++;
                Path target = readme.getParent().resolve(link.split("#")[0]);
                assertTrue(Files.exists(target), readme + " links to " + link + ", which does not exist");
            }
        }
        assertTrue(links > 200, "found only " + links + " links");
    }

    @Test
    void everyCommandInEveryReadmeIsOneWeSupport() throws IOException {
        Pattern allowed = Pattern.compile("mvn (--version|test|test -Pall|clean compile exec:java( -P[\\w-]+)?)");
        for (Path readme : readmes()) {
            String text = Files.readString(readme);
            assertFalse(text.contains("mvnw") || text.contains("exec:exec"), readme + " mentions a command that no longer exists");
            for (String line : text.split("\\R")) {
                if (line.startsWith("mvn ")) {
                    assertTrue(allowed.matcher(line).matches(), readme + ": unexpected command: " + line);
                }
            }
        }
    }

    @Test
    void theEndOfTheRootReadmeHasTheLinksAndTheEditions() throws IOException {
        String readme = read("README.md");
        int learnMore = readme.indexOf("## Learn more and get Deep Netts");
        assertTrue(learnMore > 0, "the root README has no 'Learn more and get Deep Netts' section");
        String end = readme.substring(learnMore);
        assertTrue(end.contains("https://www.deepnetts.com/cookbook/getting-started/"), "link to getting started is missing");
        assertTrue(end.contains("https://www.deepnetts.com/download/"), "link to the download page is missing");
        assertTrue(end.contains("Open source and the free option"), "the open source and free option explanation is missing");
        assertTrue(end.contains("Vector API"), "the Vector API is not explained");
        assertTrue(end.contains("CUDA"), "the GPU support is not explained");
        assertTrue(end.contains("<artifactId>deepnetts-core</artifactId>"), "the Maven dependency is missing");
        assertEquals(-1, end.indexOf("\n## "), "the section 'Learn more and get Deep Netts' must be the last one");
    }
}
