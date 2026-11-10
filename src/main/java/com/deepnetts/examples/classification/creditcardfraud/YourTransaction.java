package com.deepnetts.examples.classification.creditcardfraud;

import com.deepnetts.examples.classification.creditcardfraud.dto.Transaction;
import com.deepnetts.examples.classification.creditcardfraud.enums.MerchantCategory;

// This is the transaction the network judges.
// You can create your own transaction and see what the network thinks:
// 1. Change the values (or try the example at the bottom).
// 2. Run CreditCardFraud again.
// 3. Look at the output.
public final class YourTransaction {

    private YourTransaction() {
    }

    public static Transaction create() {
        // Suspicious: big online purchase, 3 AM, far from home.
        return new Transaction(
                420.0,                          // amount
                MerchantCategory.SHOPPING_NET,  // merchant category (see MerchantCategory for all options)
                3,                              // hour of day, 0-23
                180.0);                         // distance from home in km

        // Let's try the opposite: a small grocery purchase, 2 PM, close to home.
        // To try it, comment out the suspicious one above and uncomment the line below.
        //return new Transaction(35.0, MerchantCategory.GROCERY_POS, 14, 4.0);
    }
}
