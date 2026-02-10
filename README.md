# 🛒 SmartCart: Advanced Customer Segmentation Dashboard

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B)
![Machine Learning](https://img.shields.io/badge/Algorithms-KMeans%20%26%20Agglomerative-green)
![Status](https://img.shields.io/badge/Status-Deployed-success)

## 📌 Project Overview
**SmartCart** is an Intelligent Customer Segmentation Dashboard designed to transform raw e-commerce data into actionable marketing intelligence. By moving beyond generic strategies, this system uses unsupervised machine learning to uncover hidden patterns in customer behavior.

Unlike static analysis, this tool provides an **interactive interface** where stakeholders can compare different clustering algorithms (K-Means vs. Agglomerative) and visualize high-dimensional data in 3D.

## 🌟 Key Features
* **Dual-Model Comparison:** Compare K-Means and Agglomerative Clustering side-by-side.
* **Interactive 3D Visuals:** Explore clusters in 3D space using PCA (Principal Component Analysis).
* **Automated Optimal K Discovery:** Built-in Elbow Method and Silhouette Analysis using `KneeLocator`.
* **Segment Profiling Tab:** * **Radar Charts:** Visualize the "DNA" of each segment across all normalized metrics.
    * **Plotly Heatmaps:** Spot high-value traits across clusters with overlaid raw data annotations.
    * **Distribution Tracking:** Monitor the size of each customer group with interactive bar charts.

## 🛠️ Tech Stack
* **Frontend/App:** Streamlit
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Plotly (Interactive), Seaborn, Matplotlib
* **Machine Learning:** Scikit-Learn (K-Means, Agglomerative, PCA, StandardScaler, OneHotEncoder)
* **Optimization:** Kneed (Automated Elbow Detection)

## 📊 Methodology

### 1. Data Engineering & Cleaning
* **Imputation:** Handled missing `Income` values via median imputation.
* **Outlier Control:** Filtered illogical records (Age > 90, Income > $600k).
* **Feature Extraction:** * `Age`: Derived from birth year relative to 2026.
    * `Total Spending`: Aggregated across all product categories (Wines, Fruits, Meat, etc.).
    * `Tenure`: Calculated customer loyalty duration from the enrollment date.
    * `Family Structure`: Simplified marital status and mapped education levels for cleaner encoding.

### 2. Dimensionality Reduction
To visualize 20+ attributes in a 3D dashboard, we utilized **PCA (Principal Component Analysis)** to reduce the feature set to three principal components that capture the maximum variance in customer behavior.

### 3. Clustering Logic
* **K-Means:** Partitioning based on centroids.
* **Agglomerative:** Hierarchical clustering using 'ward' linkage to identify nested segments.



## 🚀 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/smartcart-clustering.git](https://github.com/yourusername/smartcart-clustering.git)
   cd smartcart-clustering
