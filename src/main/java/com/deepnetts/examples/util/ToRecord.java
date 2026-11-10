package com.deepnetts.examples.util;

// Converts one CSV row into a record.
public interface ToRecord<R extends Record> {
    R toRecord(String[] columns);
}
