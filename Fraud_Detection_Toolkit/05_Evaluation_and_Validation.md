# 05. Evaluation & Validation Metrics 📊

Evaluating fraud detection models is tricky because of the **Accuracy Paradox**.
If 99.9% of transactions are legitimate, a model that predicts "Legitimate" for *everything* has 99.9% accuracy—but it is useless.

## 1. The Confusion Matrix

| | Predicted: Fraud | Predicted: Legitimate |
| :--- | :--- | :--- |
| **Actual: Fraud** | **True Positive (TP)** <br> (Caught the fraud!) | **False Negative (FN)** <br> (Missed fraud - Costly!) |
| **Actual: Legitimate** | **False Positive (FP)** <br> (False Alarm - Annoying) | **True Negative (TN)** <br> (Correctly ignored) |

## 2. Key Metrics

### **Precision**
*   **Formula:** $TP / (TP + FP)$
*   **Meaning:** Of all the transactions we flagged as fraud, how many were actually fraud?
*   **Business Impact:** Low precision means you are blocking too many good customers (high False Positive Rate).

### **Recall (Sensitivity)**
*   **Formula:** $TP / (TP + FN)$
*   **Meaning:** Of all the actual fraud cases, how many did we catch?
*   **Business Impact:** Low recall means you are losing money to fraud (high False Negative Rate).

### **F1-Score**
*   **Formula:** $2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$
*   **Meaning:** Harmonic mean of Precision and Recall. Good balance for imbalanced data.

### **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)**
*   **Meaning:** Measures the ability of the model to distinguish between classes at various threshold settings.
*   **Score:** 0.5 = Random Guessing, 1.0 = Perfect.
*   **Note:** Can be misleading for highly imbalanced datasets.

### **PR-AUC (Precision-Recall Area Under Curve)**
*   **Recommendation:** **Use this instead of ROC-AUC for fraud.**
*   **Why?** It focuses specifically on the minority class (Fraud). It doesn't care about True Negatives (which dominate the dataset).

## 3. Advanced Metrics

### **Cohen's Kappa**
*   Measures inter-rater agreement. It tells you how much better your classifier is performing over the performance of a classifier that simply guesses at random according to the frequency of each class.

### **Matthews Correlation Coefficient (MCC)**
*   Regarded as one of the best single-value metrics for imbalanced classes. It takes into account TP, TN, FP, FN.
*   Range: -1 to +1.

## 4. Validation Strategies

### **Time-Series Split (Walk-Forward Validation)**
*   **Problem:** You cannot use K-Fold Cross-Validation randomly on time-series data. You cannot train on "future" data to predict "past" fraud.
*   **Solution:**
    *   Train: Jan-Mar, Test: Apr
    *   Train: Jan-Apr, Test: May
    *   Train: Jan-May, Test: Jun

### **Cost-Sensitive Learning**
*   Not all errors are equal.
*   **Cost Matrix:**
    *   Cost of False Negative (Missed Fraud) = $1000 (Loss of funds)
    *   Cost of False Positive (Blocked User) = $10 (Customer insult / support call)
*   **Goal:** Minimize Total Cost, not just maximize Accuracy.

---
**Final Advice:**
Always optimize for the business metric. Ask stakeholders: "Is it worse to lose money to fraud, or to block a legitimate user?"
