package com.deepnetts.examples;

import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;
import org.junit.jupiter.api.parallel.ResourceLock;
import org.junit.jupiter.api.parallel.Resources;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.lang.reflect.RecordComponent;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

// The validators of all examples, checked with the input that every example ships in its Your... file:
// that input is valid, a missing input is rejected with a clear message, and the warning for an input outside
// the range the network learned from appears only for such an input.
class DefaultInputsTest {

    // A validator together with the default input (YourX.create()) of its example.
    private record Pair(Class<?> validator, Object input) {
    }

    private static List<Pair> pairs() throws Exception {
        List<Class<?>> classes = ClassScan.all();
        List<Pair> pairs = new ArrayList<>();
        for (Class<?> validator : classes) {
            if (!validator.getSimpleName().endsWith("Validator")) {
                continue;
            }
            String examplePackage = validator.getPackageName().replace(".pipeline", "");
            for (Class<?> your : classes) {
                if (your.getPackageName().equals(examplePackage) && your.getSimpleName().startsWith("Your")) {
                    try {
                        pairs.add(new Pair(validator, your.getMethod("create").invoke(null)));
                    } catch (NoSuchMethodException e) {
                        // an example whose input is not a record (an image path): it has no validator of this kind
                    }
                }
            }
        }
        return pairs;
    }

    // The static method with this name that takes the input (and then the name of its source).
    private static Method method(Class<?> validator, String name, Object input) {
        return Stream.of(validator.getMethods())
                .filter(m -> m.getName().equals(name) && Modifier.isStatic(m.getModifiers()))
                .filter(m -> m.getParameterCount() >= 1 && m.getParameterTypes()[0].isInstance(input))
                .filter(m -> m.getParameterCount() == 1 || (m.getParameterCount() == 2 && m.getParameterTypes()[1] == String.class))
                .findFirst().orElse(null);
    }

    private static void call(Method method, Object input) throws Exception {
        if (method.getParameterCount() == 1) {
            method.invoke(null, input);
        } else {
            method.invoke(null, input, "test");
        }
    }

    @TestFactory
    Stream<DynamicTest> theDefaultInputOfEveryExampleIsValid() throws Exception {
        List<Pair> pairs = pairs().stream().filter(p -> method(p.validator(), "validate", p.input()) != null).toList();
        assertTrue(pairs.size() >= 10, "found only " + pairs.size() + " validators");
        return pairs.stream().map(p -> dynamicTest(p.validator().getSimpleName() + " accepts " + p.input().getClass().getSimpleName(),
                () -> assertDoesNotThrow(() -> call(method(p.validator(), "validate", p.input()), p.input()))));
    }

    @TestFactory
    Stream<DynamicTest> aMissingInputIsRejectedWithAClearMessage() throws Exception {
        List<Pair> pairs = pairs().stream().filter(p -> method(p.validator(), "validate", p.input()) != null).toList();
        return pairs.stream().map(p -> dynamicTest(p.validator().getSimpleName() + " rejects null", () -> {
            Method validate = method(p.validator(), "validate", p.input());
            InvocationTargetException e = assertThrows(InvocationTargetException.class, () -> call(validate, null));
            assertInstanceOf(IllegalArgumentException.class, e.getCause());
            assertTrue(e.getCause().getMessage().contains("must not be null"), e.getCause().getMessage());
        }));
    }

    @TestFactory
    @ResourceLock(Resources.SYSTEM_OUT)
    Stream<DynamicTest> theWarningAppearsOnlyForAnInputOutsideTheTrainingRange() throws Exception {
        List<Pair> pairs = pairs().stream().filter(p -> method(p.validator(), "warnIfOutsideData", p.input()) != null).toList();
        assertTrue(pairs.size() >= 4, "found only " + pairs.size() + " validators with the warning");
        return pairs.stream().map(p -> dynamicTest(p.validator().getSimpleName() + " warns only outside the range", () -> {
            Method warn = method(p.validator(), "warnIfOutsideData", p.input());
            assertFalse(printed(warn, p.input()).contains("Warning:"), "the default input must not warn");
            assertTrue(printed(warn, scaledUp(p.input())).contains("Warning:"), "an input 1000 times larger must warn");
        }));
    }

    // The same input with every number 1000 times larger: far above anything in the data.
    private static Object scaledUp(Object record) throws Exception {
        RecordComponent[] components = record.getClass().getRecordComponents();
        Class<?>[] types = new Class<?>[components.length];
        Object[] values = new Object[components.length];
        for (int i = 0; i < components.length; i++) {
            types[i] = components[i].getType();
            Object value = components[i].getAccessor().invoke(record);
            values[i] = value instanceof Double d ? (Object) (d * 1000) : value instanceof Integer n ? (Object) (n * 1000) : value;
        }
        return record.getClass().getDeclaredConstructor(types).newInstance(values);
    }

    private static String printed(Method method, Object input) throws Exception {
        PrintStream console = System.out;
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        try {
            System.setOut(new PrintStream(output, true, StandardCharsets.UTF_8));
            call(method, input);
        } finally {
            System.setOut(console);
        }
        return output.toString(StandardCharsets.UTF_8);
    }
}
