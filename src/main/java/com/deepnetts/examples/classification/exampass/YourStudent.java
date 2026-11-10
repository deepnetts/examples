package com.deepnetts.examples.classification.exampass;

import com.deepnetts.examples.classification.exampass.dto.Student;

// This is the student the network judges.
// You can create your own student and see what the network thinks:
// 1. Change the values (or try the example at the bottom).
// 2. Run ExamPass again.
// 3. Look at the output.
public final class YourStudent {

    private YourStudent() {
    }

    public static Student create() {
        // A student who studied 20 hours and did 3 practice tests.
        return new Student(
                20.0,   // study hours
                3);     // practice tests

        // Let's try the same student with 4 more practice tests and 12 hours less study.
        // To try it, comment out the student above and uncomment the line below.
        //return new Student(8.0, 7);
    }
}
