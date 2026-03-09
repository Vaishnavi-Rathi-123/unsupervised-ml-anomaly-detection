# 📉 Dimensionality Reduction & Anomaly Detection

## 📌 Project Overview

This project explores key machine learning concepts related to **dimensionality reduction** and **anomaly detection**.

High-dimensional datasets often contain redundant or irrelevant information that increases computational complexity and may reduce model performance. Dimensionality reduction techniques help simplify datasets while preserving important patterns.

In addition, this project demonstrates multiple approaches for detecting **anomalies (outliers)** — data points that significantly differ from normal observations.

### Techniques Covered

- **Principal Component Analysis (PCA)** – Dimensionality Reduction  
- **DBSCAN** – Density-based anomaly detection  
- **Isolation Forest** – Tree-based anomaly detection  
- **Local Outlier Factor (LOF)** – Density-based anomaly detection  

The goal is to understand both the **intuition** and **mathematical foundations** behind these methods and how they are used in practical machine learning workflows.

---

# 📂 Dimensionality Reduction

Dimensionality reduction is the process of **reducing the number of features** in a dataset while retaining the most important information.

High-dimensional data often leads to problems such as:

- Increased computational cost  
- Difficulty in visualization  
- Risk of overfitting  
- Presence of redundant or noisy features  

Reducing dimensions improves **model efficiency, interpretability, and visualization**.

### Common Dimensionality Reduction Techniques

- PCA
- t-SNE
- UMAP
- Autoencoders

In this project, the primary focus is on **Principal Component Analysis (PCA)**.

---

# 📊 Principal Component Analysis (PCA)

## Intuition

PCA transforms the original dataset into a **new coordinate system** where:

- **PC1** captures the maximum variance in the data  
- **PC2** captures the second highest variance and is orthogonal to PC1  

This allows the dataset to be represented in **fewer dimensions** while preserving most of the variability.

```
PC1 → Maximum variance direction  
PC2 → Second highest variance (orthogonal to PC1)
```

---

## Mathematical Steps of PCA

### 1️⃣ Standardize the Data

Before applying PCA, the data must be standardized so that each feature has:

- Mean = 0  
- Standard Deviation = 1  

```
z = (x − μ) / σ
```

---

### 2️⃣ Compute the Covariance Matrix

The covariance matrix measures how variables vary together.

```
C = (1 / n) XᵀX
```

---

### 3️⃣ Compute Eigenvalues and Eigenvectors

- **Eigenvectors** represent the principal directions  
- **Eigenvalues** represent the amount of variance captured by each component

```
Cv = λv
```

Where:

- **λ** → eigenvalue  
- **v** → eigenvector  

---

### 4️⃣ Select Principal Components

Principal components are sorted by eigenvalues.

```
PC1 > PC2 > PC3
```

The components with the **largest eigenvalues** retain the most information.

---

## PCA Optimization Perspective

PCA can be interpreted in two equivalent ways:

### Minimizing Reconstruction Error

```
minimize Σ(xᵢ − x̂ᵢ)²
```

### Maximizing Variance

```
maximize Σ(xᵢ − x̄)²
```

---

# 🚨 Anomaly Detection

Anomaly detection identifies **unusual observations** that differ significantly from the majority of the data.

Anomalies may indicate:

- Fraudulent transactions
- Network intrusions
- System failures
- Medical abnormalities

Example representation:

```
Normal data → ● ● ● ● ●
Normal data → ● ● ● ●
Anomaly     → ★
```

Applications include:

- Finance
- Cybersecurity
- Healthcare
- Manufacturing systems

---

# 🔍 DBSCAN for Anomaly Detection

**DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** groups data points based on density.

Points that do not belong to any cluster are classified as **noise**, which may represent anomalies.

---

## Key Parameters

### ε (Epsilon)

Maximum distance between two points to be considered neighbors.

### MinPts (Minimum Points)

Minimum number of points required to form a dense region.

---

## Choosing Parameters

### Selecting ε

A common approach is the **k-distance graph**.

Steps:

1. Compute distance to the k-th nearest neighbor  
2. Plot distances  
3. Select the **elbow point** as epsilon  

---

### Selecting MinPts

A common rule:

```
min_samples = 2 × number_of_features
```

---

## DBSCAN Output

- **Cluster points → Normal data**
- **Noise points → Anomalies**

---

# 🌳 Isolation Forest for Anomaly Detection

Isolation Forest is a **tree-based algorithm** designed specifically for anomaly detection.

### Key Idea

Anomalies are easier to isolate than normal data points.

```
Anomalies → isolated quickly  
Normal points → require many splits
```

---

## How Isolation Forest Works

1. Randomly select a feature  
2. Randomly choose a split value  
3. Partition the data  
4. Repeat until data points are isolated  

Multiple trees are built to form an **Isolation Forest**.

---

## Path Length Concept

The algorithm measures how many splits are needed to isolate a point.

- **Short path length → anomaly**
- **Long path length → normal data**

---

# 📊 Local Outlier Factor (LOF)

LOF detects anomalies by comparing the **density of a data point** with the density of its neighbors.

### Core Idea

An anomaly has **much lower local density** than its neighbors.

---

## LOF Algorithm Steps

### 1️⃣ Find k-Nearest Neighbors

```
k = number of neighbors
```

### 2️⃣ Compute Local Density

Density is estimated using the distance between a point and its neighbors.

### 3️⃣ Compare Density

```
LOF = density of neighbors / density of point
```

---

## LOF Score Interpretation

| LOF Score | Interpretation |
|----------|---------------|
| ≈ 1 | Normal point |
| > 1 | Possible anomaly |
| >> 1 | Strong anomaly |
| < 1 | Point inside dense cluster |

### Example

| Point | LOF Score | Interpretation |
|------|-----------|---------------|
| A | 0.9 | Dense cluster |
| B | 1.0 | Normal point |
| C | 1.6 | Potential anomaly |
| D | 3.2 | Strong anomaly |

---

# 📈 Key Observations

| Method | Purpose | Key Idea |
|------|------|------|
| PCA | Dimensionality Reduction | Capture maximum variance |
| DBSCAN | Clustering / Anomaly Detection | Noise points represent anomalies |
| Isolation Forest | Anomaly Detection | Anomalies isolate quickly |
| LOF | Anomaly Detection | Compare local density |

---

# 🛠 Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

# 🎯 Conclusion

Dimensionality reduction and anomaly detection are essential components of modern machine learning workflows.

### Key Insights

- **PCA** reduces dimensionality by preserving maximum variance.
- **DBSCAN** identifies clusters and automatically detects noise points.
- **Isolation Forest** isolates anomalous observations using random trees.
- **LOF** detects anomalies by comparing local density differences.

These techniques are widely used in:

- Fraud detection
- Cybersecurity
- System monitoring
- Data preprocessing pipelines

---

# 👩‍💻 Author

**Vaishnavi Rathi**

---
