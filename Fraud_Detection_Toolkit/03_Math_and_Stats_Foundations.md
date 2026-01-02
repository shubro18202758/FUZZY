# 03. Mathematical & Statistical Foundations 📐

Understanding the math behind the models is crucial for interpreting results and debugging issues.

## 1. Probability Distributions

### **Normal Distribution (Gaussian)**
*   **Concept:** The "Bell Curve". Most natural phenomena follow this.
*   **Relevance:** Many statistical outlier detection methods (like Z-Score) assume data is normally distributed.
*   **Reality Check:** Financial data is rarely normal. It is usually **highly skewed** (Power Law).

### **Power Law / Pareto Distribution**
*   **Concept:** The "80/20 Rule". A small number of transactions account for the vast majority of the volume (or fraud).
*   **Relevance:** Wealth distribution, transaction amounts, and network degree distributions often follow a Power Law.

### **Poisson Distribution**
*   **Concept:** Models the number of events occurring in a fixed interval of time.
*   **Relevance:** Modeling the *count* of transactions per hour. If a user typically makes 2 transactions/day (Poisson $\lambda=2$), and suddenly makes 50, the probability of this happening by chance is infinitesimally small.

## 2. Descriptive Statistics

*   **Mean ($\mu$):** Average. Sensitive to outliers.
*   **Median:** Middle value. Robust to outliers.
*   **Standard Deviation ($\sigma$):** Measure of spread.
*   **Skewness:** Measure of asymmetry. Financial data is usually right-skewed (long tail of high amounts).
*   **Kurtosis:** Measure of "tailedness". High kurtosis indicates frequent extreme outliers.

## 3. Outlier Detection Methods

### **Z-Score (Standard Score)**
*   **Formula:** $Z = \frac{x - \mu}{\sigma}$
*   **Logic:** How many standard deviations is $x$ away from the mean?
*   **Threshold:** Typically, $|Z| > 3$ is considered an outlier.
*   **Limitation:** Assumes Normal distribution.

### **IQR (Interquartile Range)**
*   **Logic:** Uses quartiles (Q1 = 25th percentile, Q3 = 75th percentile).
*   **Formula:** $IQR = Q3 - Q1$.
*   **Bounds:**
    *   Lower Bound: $Q1 - 1.5 \times IQR$
    *   Upper Bound: $Q3 + 1.5 \times IQR$
*   **Pros:** Robust to extreme outliers compared to Z-Score.

## 4. Benford's Law (The First-Digit Law)
*   **Concept:** In many naturally occurring collections of numbers, the leading digit is likely to be small.
    *   1 appears ~30.1% of the time.
    *   9 appears ~4.6% of the time.
*   **Application:** Detecting fabricated financial figures. If a fraudster invents transaction amounts, they often distribute digits uniformly (Random 1-9), violating Benford's Law.

## 5. Bayesian Probability
*   **Bayes' Theorem:** $P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$
*   **Application:** Updating the probability of fraud given new evidence.
    *   $P(Fraud | HighAmount)$
    *   We start with a prior probability (base rate of fraud), and update it when we see a "High Amount".

## 6. Distance Metrics
Used in clustering (K-Means) and Nearest Neighbors (KNN).

*   **Euclidean Distance:** Straight line distance. Sensitive to scale (requires normalization).
    *   $d(p, q) = \sqrt{\sum (p_i - q_i)^2}$
*   **Manhattan Distance:** Sum of absolute differences. Better for high dimensions.
    *   $d(p, q) = \sum |p_i - q_i|$
*   **Mahalanobis Distance:**
    *   Measures distance from a point to a distribution.
    *   **Crucial for Anomaly Detection:** It accounts for the correlation between variables. A point might be within reasonable ranges for X and Y individually, but their *combination* is impossible (e.g., High Age + Low Income might be fine, Low Age + High Income might be fine, but specific correlations matter).

---
**Key Takeaway:**
Fraud detection is essentially finding data points that have a **low probability** of occurring under the "Normal" distribution of user behavior.
