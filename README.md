# 🛒 SmartCart Customer Segmentation System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-KMeans-green)
![Status](https://img.shields.io/badge/Status-Completed-orange)

## 📌 Project Overview

**SmartCart** is a growing e-commerce platform with a global customer base. Currently, the company uses generic marketing strategies that fail to address the specific needs of different user groups, leading to inefficient marketing and potential customer churn.

This project implements an **Intelligent Customer Segmentation System** using unsupervised machine learning. By analyzing customer demographics, purchase history, and website activity, we cluster users into distinct segments. This allows the marketing team to deliver personalized campaigns, improve customer retention, and identify high-value shoppers.

## 📂 Dataset Description

The dataset comprises **2,240 customer records** with **22 attributes**.

| Feature Category | Key Attributes |
| :--- | :--- |
| **Demographics** | `Year_Birth`, `Education`, `Marital_Status`, `Income`, `Kidhome`, `Teenhome` |
| **Purchase History** | `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds` |
| **Engagement** | `NumWebPurchases`, `NumStorePurchases`, `NumDealsPurchases`, `NumWebVisitsMonth` |
| **Loyalty** | `Dt_Customer` (Enrollment Date), `Recency` (Days since last purchase) |

## 🛠️ Tech Stack

* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (K-Means Clustering)
* **Environment:** Jupyter Notebook

## 📊 Methodology

1.  **Data Preprocessing**:
    * **Missing Values:** Imputed missing `Income` values using the median.
    * **Date Handling:** Converted `Dt_Customer` to datetime format.
    * **Outlier Removal:** Filtered illogical records (e.g., Age > 90, Income > 600k).
2.  **Feature Engineering**:
    * **Age**: Calculated from `Year_Birth`.
    * **Customer Tenure**: Derived from `Dt_Customer` relative to the most recent dataset date.
    * **Total Spending**: Aggregated spending across all product categories.
    * **Total Children**: Combined `Kidhome` and `Teenhome`.
    * **Family Structure**: Simplified `Marital_Status` into "Partner" vs. "Alone" and mapped `Education` levels.
3.  **Exploratory Data Analysis (EDA)**:
    * **Correlation Matrix:** Visualized relationships between income and spending habits.
    * **Pairplots:** Analyzed distributions and potential cluster separations.
4.  **Clustering**:
    * Applied **K-Means Clustering** to group customers based on spending and engagement patterns.

## 🚀 How to Run

1.  Clone the repository:
    ```bash
    git clone [https://github.com/yourusername/smartcart-clustering.git](https://github.com/yourusername/smartcart-clustering.git)
    ```
2.  Install dependencies:
    ```bash
    pip install pandas matplotlib seaborn scikit-learn
    ```
3.  Open the notebook:
    ```bash
    jupyter notebook smart_cart.ipynb
    ```

## 🔮 Future Scope

* Integration of a recommendation engine for specific clusters.
* Deployment of a dashboard using Streamlit for real-time customer analysis.
