package com.deepnetts.examples;

import com.deepnetts.examples.util.CsvEncoder;

import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.TestFactory;
import org.junit.jupiter.api.parallel.ResourceLock;
import org.junit.jupiter.api.parallel.Resources;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.net.URISyntaxException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

// Runs every example, from the project root like a user would, twice, in a language with a decimal comma.
// That takes about 10 minutes, so it runs only with mvn test -Pall (the "slow" tests).
// It finds the examples itself (a class with a static main(String[]) and a static run()), so a new example is tested
// without adding it here. The examples write models/, which is ignored by git.
class ExamplesTest {

    // A decimal comma between two digits, such as "76,6": the examples print numbers with a decimal point.
    private static final Pattern DECIMAL_COMMA = Pattern.compile("\\d,\\d");

    // Any character outside plain ASCII, such as a dash or R². No Windows code page shows them all: the console prints "?".
    private static final Pattern NOT_ASCII = Pattern.compile("[^\\x00-\\x7F]");

    // Examples these tests don't run: VGG16 downloads 600 MB and needs about 4 GB of free memory. Run it by hand.
    private static final Set<String> NOT_RUN_BY_TESTS = Set.of("Vgg16Import");

    // Examples whose result is not compared with the baseline: Hot dog does not learn reliably (see its README),
    // so its small margin over the baseline would make the test fail on another computer.
    private static final Set<String> NOT_CHECKED_AGAINST_BASELINE = Set.of("HotDog");

    @Tag("slow")
    @TestFactory
    @ResourceLock(Resources.SYSTEM_OUT)
    @ResourceLock(Resources.LOCALE)
    Stream<DynamicTest> everyExampleRunsAndGivesTheSameResultsEveryTime() throws IOException, URISyntaxException {
        List<Class<?>> examples = findExamples().stream()
                .filter(example -> !NOT_RUN_BY_TESTS.contains(example.getSimpleName()))
                .toList();
        assertTrue(examples.size() >= 17, "found only " + examples.size() + " examples");

        return examples.stream().map(example -> dynamicTest(example.getSimpleName(), () -> {
            String firstRun = run(example);
            String secondRun = run(example);

            assertTrue(firstRun.toLowerCase(Locale.ROOT).contains("stopped after"), firstRun);
            assertFalse(firstRun.contains("Exception"), firstRun);
            assertFalse(firstRun.contains("Warning:"), "the default Your... input is outside the training range:\n" + firstRun);
            assertFalse(DECIMAL_COMMA.matcher(firstRun).find(), "a decimal comma in:\n" + firstRun);
            assertFalse(NOT_ASCII.matcher(firstRun).find(), "a character a Windows console shows as ?, in:\n" + firstRun);
            assertEquals(firstRun, secondRun, "the same SEED must give the same output");
            if (!NOT_CHECKED_AGAINST_BASELINE.contains(example.getSimpleName())) {
                assertEquals("", OutputChecks.problemWithBaseline(firstRun), "the network must beat the baseline:\n" + firstRun);
            }
        }));
    }

    // Examples that divide each input by "the largest value in the data" (the MAX_... constants in FeatureExtractor):
    // every row of their CSV file must give inputs from 0 to 1. Fails when a dataset changes but a constant doesn't.
    @TestFactory
    Stream<DynamicTest> everyCsvRowGivesInputsFrom0To1() throws IOException, URISyntaxException {
        List<Class<?>> classes = findClasses();
        return findExamples().stream().filter(ExamplesTest::scalesByConstants).map(example ->
                dynamicTest(example.getSimpleName(), () -> {
                    String pipeline = example.getPackageName() + ".pipeline.";
                    // The example's encoder: the class in its pipeline folder that reads the CSV file.
                    Class<?> encoderClass = classes.stream()
                            .filter(type -> type.getName().startsWith(pipeline) && CsvEncoder.class.isAssignableFrom(type))
                            .findFirst().orElseThrow();
                    CsvEncoder<?> encoder = (CsvEncoder<?>) encoderClass.getDeclaredConstructor().newInstance();
                    Class<?> extractor = load(pipeline + "FeatureExtractor");
                    Method extract = Stream.of(extractor.getMethods())
                            .filter(m -> m.getName().equals("extract") && m.getParameterCount() == 1)
                            .findFirst().orElseThrow();
                    @SuppressWarnings("unchecked")
                    List<String> names = (List<String>) extractor.getField("FEATURE_NAMES").get(null);

                    Field datasetField = example.getDeclaredField("DATASET");
                    datasetField.setAccessible(true);
                    String dataset = (String) datasetField.get(null);
                    int[] row = {0};
                    encoder.forEachRow(dataset, columns -> {
                        row[0]++;
                        float[] inputs;
                        try {
                            inputs = (float[]) extract.invoke(null, encoder.toRecord(columns));
                        } catch (ReflectiveOperationException e) {
                            throw new IllegalStateException(e);
                        }
                        for (int i = 0; i < inputs.length; i++) {
                            String where = dataset + ", data row " + row[0] + ", " + names.get(i) + " = " + inputs[i];
                            assertTrue(inputs[i] >= -1e-6f && inputs[i] <= 1 + 1e-6f, where);
                        }
                    });
                }));
    }

    private static boolean scalesByConstants(Class<?> example) {
        try {
            Class<?> extractor = Class.forName(example.getPackageName() + ".pipeline.FeatureExtractor");
            return Stream.of(extractor.getDeclaredFields()).anyMatch(field -> field.getName().startsWith("MAX_"));
        } catch (ClassNotFoundException e) {
            return false;
        }
    }

    private static List<Class<?>> findExamples() throws IOException, URISyntaxException {
        return findClasses().stream().filter(ExamplesTest::isExample).toList();
    }

    // Every class of the examples (in target/classes), sorted by name.
    private static List<Class<?>> findClasses() throws IOException, URISyntaxException {
        Path root = Path.of(ExamplesTest.class.getProtectionDomain().getCodeSource().getLocation().toURI())
                .resolveSibling("classes");   // target/classes next to target/test-classes
        try (Stream<Path> files = Files.walk(root)) {
            return files.filter(file -> file.toString().endsWith(".class") && !file.getFileName().toString().contains("$"))
                    .map(file -> root.relativize(file).toString().replace('/', '.').replace('\\', '.'))
                    .map(name -> name.substring(0, name.length() - ".class".length()))
                    .map(ExamplesTest::load)
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

    private static boolean isExample(Class<?> type) {
        try {
            return Modifier.isStatic(type.getDeclaredMethod("main", String[].class).getModifiers())
                    && isStaticNoArgs(type.getDeclaredMethod("run"));
        } catch (NoSuchMethodException e) {
            return false;
        }
    }

    private static boolean isStaticNoArgs(Method method) {
        return Modifier.isStatic(method.getModifiers()) && method.getParameterCount() == 0;
    }

    private static String run(Class<?> example) throws Exception {
        PrintStream console = System.out;
        Locale defaultLocale = Locale.getDefault();
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        try {
            Locale.setDefault(Locale.of("sr", "RS"));   // a language with a decimal comma
            System.setOut(new PrintStream(output, true, StandardCharsets.UTF_8));
            Method run = example.getDeclaredMethod("run");
            run.setAccessible(true);
            run.invoke(null);
        } finally {
            System.setOut(console);
            Locale.setDefault(defaultLocale);
        }
        return output.toString(StandardCharsets.UTF_8);
    }
}
