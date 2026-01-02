# 04. Anomaly Detection Algorithms 🤖

There are three main approaches to fraud detection:
1.  **Supervised Learning:** We have labels (Fraud / Not Fraud).
2.  **Unsupervised Learning:** We have no labels; we look for weird patterns.
3.  **Semi-Supervised:** We train only on "Normal" data and flag anything that deviates.

## 1. Statistical & Machine Learning (Supervised)

### **Logistic Regression**
*   **Type:** Linear Model.
*   **Pros:** Highly interpretable (coefficients tell you risk factors). Fast.
*   **Cons:** Cannot capture complex, non-linear relationships.

### **Random Forest / Gradient Boosting (XGBoost, CatBoost)**
*   **Type:** Ensemble of Decision Trees.
*   **Pros:** State-of-the-art for tabular data. Handles non-linearities and interactions automatically.
*   **Cons:** Can overfit if not regularized. Black-box nature (requires SHAP values for explainability).

## 2. Unsupervised Anomaly Detection

### **Isolation Forest (iForest)**
*   **Concept:** It is easier to "isolate" an anomaly than a normal point.
*   **Mechanism:** Randomly selects a feature and randomly selects a split value. Anomalies require fewer splits to be isolated (shorter path length in the tree).
*   **Pros:** Very efficient, works well with high-dimensional data. No distance calculation needed.

### **Local Outlier Factor (LOF)**
*   **Concept:** Density-based.
*   **Mechanism:** Compares the local density of a point to the local densities of its neighbors. If a point has a much lower density than its neighbors, it is an outlier.
*   **Pros:** Good for detecting local outliers (anomalies relative to their cluster).

### **DBSCAN (Density-Based Spatial Clustering)**
*   **Concept:** Clustering.
*   **Mechanism:** Groups points that are closely packed together. Points in low-density regions are marked as noise (outliers).
*   **Pros:** Can find arbitrarily shaped clusters.

## 3. Deep Learning Approaches

### **Autoencoders (Reconstruction-based)**
*   **Architecture:** Input Layer -> Encoder -> Bottleneck (Compressed Representation) -> Decoder -> Output Layer.
*   **Training:** Train **only on legitimate transactions**. The model learns to compress and reconstruct normal data perfectly.
*   **Detection:** When a fraud case is fed in, the model fails to reconstruct it accurately.
*   **Score:** `Reconstruction Error = MSE(Input, Output)`. High error = Fraud.

### **GANs (Generative Adversarial Networks)**
*   **Architecture:** Generator vs. Discriminator.
*   **AnoGAN:** Train a GAN on normal data. The Discriminator learns to distinguish "Real Normal" from "Fake". Anomalies will be flagged as "Fake" or have high residual loss.

## 4. Graph-Based Algorithms

### **Connected Components**
*   Finds disjoint subgraphs. Useful for finding isolated fraud rings.

### **Louvain Modularity**
*   Community detection algorithm. Finds clusters of highly connected nodes (e.g., a group of mule accounts sending money to each other).

---
**Algorithm Selection Guide:**

| Scenario | Recommended Algorithm |
| :--- | :--- |
| **Labeled Data Available** | XGBoost, LightGBM, CatBoost |
| **No Labels (Unsupervised)** | Isolation Forest, Autoencoders |
| **Spatial / Density Data** | DBSCAN, LOF |
| **Complex Relationships** | Graph Neural Networks (GNN) |
