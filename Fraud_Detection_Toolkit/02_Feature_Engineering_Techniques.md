# 02. Feature Engineering Techniques for Fraud Detection 🛠️

In fraud detection, **Feature Engineering is more important than the model itself**. A simple Logistic Regression with excellent features will often outperform a complex Neural Network with raw features.

This guide covers the specific techniques used to expose fraudulent patterns in financial data.

## 1. Aggregation Features (Transaction History)
Fraudsters often exhibit bursts of activity. We capture this by aggregating history.

*   **Velocity Features:**
    *   Number of transactions in the last `1 hour`, `24 hours`, `7 days`.
    *   *Example:* `count_trx_last_1h` > 10 might indicate a bot attack.
*   **Amount Aggregations:**
    *   Average/Sum/Max amount spent in the last `X` time window.
    *   *Example:* Current transaction amount vs. `avg_amount_last_30d`. A sudden spike is suspicious.
*   **Distinct Counts:**
    *   Number of distinct merchants/countries/terminals used in the last `24h`.
    *   *Example:* Using the card in 5 different countries in 1 hour.

## 2. Time-Based Features
Raw timestamps are useless. We must extract meaning.

*   **Cyclical Features:**
    *   Hour of Day (0-23), Day of Week (0-6).
    *   *Insight:* Fraud often happens during late-night hours (2 AM - 5 AM).
*   **Time Since Last Transaction:**
    *   Calculate `Time_Delta = Current_Timestamp - Last_Transaction_Timestamp`.
    *   *Insight:* Very short time deltas imply automated scripts or rapid-fire attacks.
*   **Time Since First Transaction:**
    *   Age of the account. New accounts are riskier.

## 3. Ratio & Interaction Features
Combining two features to create a stronger signal.

*   **Transaction to Average Ratio:**
    *   `Current_Amount / Avg_Amount_Last_30d`.
    *   If the ratio is > 10, it's a massive deviation from normal behavior.
*   **Credit Limit Usage:**
    *   `Current_Balance / Credit_Limit`.
    *   Maxing out cards rapidly is a fraud signal.

## 4. Geospatial Features
If location data (Lat/Long or IP address) is available.

*   **Distance Calculation (Haversine Formula):**
    *   Distance between the current transaction and the user's home address.
    *   Distance between the current transaction and the *previous* transaction.
*   **Velocity of Travel:**
    *   `Distance / Time_Delta`.
    *   *Impossible Travel:* If a user transacts in London and then New York 1 hour later, the speed required > 3000 mph. This is physically impossible.

## 5. Frequency Encoding
Handling high-cardinality categorical variables (like `Merchant_ID` or `Zip_Code`).

*   **Count Encoding:** Replace the category with the number of times it appears in the dataset.
    *   *Insight:* Rare merchants or rare zip codes might be riskier or require investigation.

## 6. Behavioral Profiling (The "User Profile")
Instead of looking at the transaction in isolation, compare it to the user's "Normal".

*   **Z-Score of Current Transaction:**
    *   $Z = \frac{X - \mu_{user}}{\sigma_{user}}$
    *   Where $\mu_{user}$ is the user's average spending and $\sigma_{user}$ is their standard deviation.
    *   A Z-score > 3 means the transaction is 3 standard deviations away from their norm.

## 7. Graph-Based Features (Network Analysis)
*   **Degree:** How many different accounts has this user sent money to?
*   **PageRank:** How "important" or central is this node in the flow of money?
*   **Cycles:** Does money flow A -> B -> C -> A? (Money laundering pattern).

---
**Example Python Snippet (Pandas):**

```python
# Rolling window aggregation
df['trx_count_1h'] = df.groupby('user_id')['amount'].transform(
    lambda x: x.rolling('1H').count()
)

# Time since last transaction
df['last_trx_time'] = df.groupby('user_id')['timestamp'].shift(1)
df['time_since_last'] = (df['timestamp'] - df['last_trx_time']).dt.seconds
```
