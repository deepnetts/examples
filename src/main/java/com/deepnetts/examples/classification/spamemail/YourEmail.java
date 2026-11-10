package com.deepnetts.examples.classification.spamemail;

import com.deepnetts.examples.classification.spamemail.dto.Email;

// This is the email the network judges.
// You can create your own email and see what the network thinks:
// 1. Change the values (or try the example at the bottom).
// 2. Run SpamEmail again.
// 3. Look at the output.
public final class YourEmail {

    private YourEmail() {
    }

    public static Email create() {
        // Suspicious: "free", "money" and "remove" words, lots of "!" and "$", long CAPITAL LETTERS.
        return new Email(
                0.5,     // "free" word, % of the email
                0.2,     // "money" word, % of the email
                0.3,     // "remove" word, % of the email
                1.4,     // "your" word, % of the email
                0.9,     // "!" characters, % of the email
                0.2,     // "$" characters, % of the email
                120,     // longest run of capital letters in a row
                600);    // total capital letters

        // Let's try the opposite: a normal work email, no spammy words, almost no capitals.
        // To try it, comment out the suspicious one above and uncomment the line below.
        //return new Email(0, 0, 0, 0.3, 0, 0, 8, 120);
    }
}
