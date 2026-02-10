import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from kneed import KneeLocator
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import plotly.express as px

# 1. Page Configuration (Must be first)
st.set_page_config(page_title="Customer Segmentation", layout="wide")


# 2. Caching Data Loading & Cleaning
# This prevents the app from reloading the CSV every time you move a slider.
@st.cache_data
def load_and_clean_data(filepath):
    # Try/Except to handle if file doesn't exist
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        st.error(f"File '{filepath}' not found. Please ensure the file is in the same directory.")
        st.stop()

    # Filling missing values
    df["Income"] = df["Income"].fillna(df["Income"].median())

    # Feature Engineering
    current_year = pd.Timestamp.now().year
    df["Age"] = current_year - df["Year_Birth"]

    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)
    reference_date = df["Dt_Customer"].max()
    df["Customer_Tenure_Days"] = (reference_date - df["Dt_Customer"]).dt.days

    df["Total_Spending"] = (
            df["MntWines"] + df["MntFruits"] + df["MntFishProducts"] +
            df["MntMeatProducts"] + df["MntGoldProds"] + df["MntSweetProducts"]
    )

    df["Total_Children"] = df["Kidhome"] + df["Teenhome"]

    # Simplified Mappings
    education_map = {
        "Basic": "Undergraduate", "2n Cycle": "Undergraduate",
        "Graduation": "Graduate", "Master": "Postgraduate", "PhD": "Postgraduate"
    }
    df["Education"] = df["Education"].replace(education_map)

    partner_map = {
        "Married": "Partner", "Together": "Partner",
        "Single": "Alone", "Divorced": "Alone", "Widow": "Alone",
        "Absurd": "Alone", "YOLO": "Alone"
    }
    df["Living_With"] = df["Marital_Status"].replace(partner_map)

    # Dropping columns and outliers
    cols_drop = ["ID", "Year_Birth", "Marital_Status", "Kidhome", "Teenhome",
                 "Dt_Customer", "MntWines", "MntFruits", "MntMeatProducts",
                 "MntGoldProds", "MntSweetProducts", "MntFishProducts",
                 "Z_CostContact", "Z_Revenue"]  # Added Z cols usually in this dataset

    # Drop only existing columns
    df_cleaned = df.drop(columns=[c for c in cols_drop if c in df.columns])
    df_cleaned = df_cleaned[(df_cleaned["Age"] < 90) & (df_cleaned["Income"] < 600000)]

    return df, df_cleaned


# 3. Caching Preprocessing & PCA
@st.cache_data
def preprocess_data(df_cleaned):
    # Encoding
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    cat_cols = ["Education", "Living_With"]
    enc_cols = ohe.fit_transform(df_cleaned[cat_cols])

    enc_df = pd.DataFrame(enc_cols, columns=ohe.get_feature_names_out(cat_cols), index=df_cleaned.index)
    df_final = pd.concat([df_cleaned.drop(columns=cat_cols), enc_df], axis=1)

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_final)

    # PCA
    pca = PCA(n_components=3)
    x_pca = pca.fit_transform(X_scaled)

    return df_final, x_pca


# 4. Caching Metric Calculations (Expensive Loop)
@st.cache_data
def calculate_metrics(x_pca):
    wcss = []
    scores = []
    k_range = range(2, 11)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(x_pca)
        wcss.append(kmeans.inertia_)
        scores.append(silhouette_score(x_pca, labels))

    return wcss, scores, k_range


# --- Main Execution Flow ---

# Load Data
df, df_cleaned = load_and_clean_data("smartcart_customers.csv")
X_final, x_pca = preprocess_data(df_cleaned)

# Header
st.title("🛒 Smart-Cart Customer Segmentation")
st.markdown("### Clustering Analysis using K-Means & Agglomerative Algorithms")
st.markdown("---")

# Tab Layout for better organization
tab1, tab2, tab3 , tab4 = st.tabs(["📊 Dataset & Features", "📉 Optimal K Analysis", "🧊 3D Clustering Model" , "🎯 Segment Profiling"])

with tab1:
    col_data, col_heat = st.columns([1, 2])

    with col_data:
        st.subheader("Raw Data Preview")
        st.dataframe(df.head(), height=400)

    with col_heat:
        st.subheader("Correlation Heatmap")
        fig_heat, ax_heat = plt.subplots(figsize=(10, 8))
        corr = df_cleaned.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", annot_kws={"size": 8}, ax=ax_heat)
        st.pyplot(fig_heat)

with tab2:
    st.subheader("Determining Optimal Clusters")

    wcss, scores, k_range = calculate_metrics(x_pca)

    # Knee Locator
    # Note: Need to insert a dummy value at start of wcss to match range(1,11) logic if needed,
    # but kneed handles arrays well.
    kl = KneeLocator(range(2, 11), wcss, curve="convex", direction="decreasing")
    optimal_k = kl.elbow

    col_metrics, col_info = st.columns([3, 1])

    with col_metrics:
        fig_opt, ax1 = plt.subplots(figsize=(10, 5))

        # WCSS Plot
        ax1.plot(k_range, wcss, marker='o', color='tab:blue', label='WCSS')
        ax1.set_xlabel("Number of Clusters (k)")
        ax1.set_ylabel("WCSS (Inertia)", color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.grid(True, alpha=0.3)

        # Silhouette Plot
        ax2 = ax1.twinx()
        ax2.plot(k_range, scores, marker='s', color='tab:red', linestyle='--', label='Silhouette Score')
        ax2.set_ylabel("Silhouette Score", color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        plt.title("Elbow Method & Silhouette Analysis")
        st.pyplot(fig_opt)

    with col_info:
        st.info(f"**Optimal K (Elbow Method):** {optimal_k}")
        st.write("The Silhouette score indicates how well-separated the clusters are. Higher is better.")

with tab3:
    st.subheader("Cluster Visualization")

    # Controls
    col_controls1, col_controls2 = st.columns(2)
    with col_controls1:
        k_val = st.slider("Select K-Means Clusters", 2, 10, int(optimal_k) if optimal_k else 4, key="k_slider")
    with col_controls2:
        agg_val = st.slider("Select Agglomerative Clusters", 2, 10, int(optimal_k) if optimal_k else 4,
                            key="agg_slider")

    # Models & Plots
    col_km, col_agg , char = st.columns(3)

    # --- K-Means Column ---
    with col_km:
        st.markdown("#### K-Means")
        kmeans = KMeans(n_clusters=k_val, random_state=42, n_init=10)
        labels_km = kmeans.fit_predict(x_pca)

        # 3D Plot
        fig_km = plt.figure(figsize=(6, 6))
        ax_km = fig_km.add_subplot(111, projection="3d")
        sc_km = ax_km.scatter(x_pca[:, 0], x_pca[:, 1], x_pca[:, 2], c=labels_km, cmap='viridis', s=40)

        # Legend Fix
        ax_km.legend(*sc_km.legend_elements(), title="Cluster", loc="upper left", bbox_to_anchor=(0, 1))

        ax_km.set_xlabel("PCA 1")
        ax_km.set_ylabel("PCA 2")
        ax_km.set_zlabel("PCA 3")
        ax_km.set_title(f"K-Means (k={k_val})")
        st.pyplot(fig_km )

    # --- Agglomerative Column ---
    with col_agg:
        st.markdown("#### Agglomerative")
        agg = AgglomerativeClustering(n_clusters=agg_val, linkage="ward")
        labels_agg = agg.fit_predict(x_pca)

        # 3D Plot
        fig_agg = plt.figure(figsize=(6, 6))
        ax_agg = fig_agg.add_subplot(111, projection="3d")
        sc_agg = ax_agg.scatter(x_pca[:, 0], x_pca[:, 1], x_pca[:, 2], c=labels_agg, cmap='plasma', s=40)

        # Legend Fix
        ax_agg.legend(*sc_agg.legend_elements(), title="Cluster", loc="upper left", bbox_to_anchor=(0, 1))

        ax_agg.set_xlabel("PCA 1")
        ax_agg.set_ylabel("PCA 2")
        ax_agg.set_zlabel("PCA 3")
        ax_agg.set_title(f"Agglomerative (k={agg_val})")
        st.pyplot(fig_agg)


    st.markdown("---")

with tab4:
    st.header("Deep Dive into Customer Segments")

    # 1. Data Preparation
    analysis_df = df_cleaned.copy()
    analysis_df["Cluster"] = labels_agg  # Using Agglomerative as per your request

    # Calculate Mean Profile
    metrics = ["Income", "Age", "Total_Spending", "Total_Children", "Customer_Tenure_Days"]
    profile = analysis_df.groupby("Cluster")[metrics].mean()

    # Normalize for Visualization (so colors/radar lines are comparable)
    scaler = MinMaxScaler()
    profile_norm = pd.DataFrame(scaler.fit_transform(profile), columns=metrics, index=profile.index)

    # --- ROW 1: Visual Profiles ---
    col_radar, col_heat = st.columns(2)

    with col_radar:
        st.subheader("🕸️ Cluster Radar Profile")
        fig_radar = go.Figure()
        for cluster_id in profile_norm.index:
            fig_radar.add_trace(go.Scatterpolar(
                r=profile_norm.loc[cluster_id],
                theta=metrics,
                fill='toself',
                name=f'Cluster {cluster_id}'
            ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_heat:
        st.subheader("🔥 Feature Intensity Heatmap")
        # Plotly Heatmap
        fig_heat = px.imshow(
            profile_norm,
            labels=dict(x="Features", y="Cluster", color="Intensity"),
            x=metrics,
            y=[f"Cluster {i}" for i in profile_norm.index],
            color_continuous_scale="RdYlGn",
            aspect="auto"
        )
        # Add text annotations to show the actual values on top of colors
        fig_heat.update_traces(text=profile.values.round(2), texttemplate="%{text}")
        st.plotly_chart(fig_heat, use_container_width=True)

    # --- ROW 2: Distribution & Data ---
    st.divider()
    col_dist, col_raw = st.columns([2, 1])

    with col_dist:
        st.subheader("👥 Cluster Distribution")
        # Plotly Countplot (Bar chart)
        counts = analysis_df["Cluster"].value_counts().reset_index()
        counts.columns = ["Cluster", "Count"]
        counts = counts.sort_values("Cluster")

        fig_bar = px.bar(
            counts,
            x="Cluster",
            y="Count",
            color="Cluster",
            text="Count",
            color_continuous_scale="plasma"
        )
        fig_bar.update_traces(textposition='outside')
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_raw:
        st.subheader("📋 Mean Values")
        st.write(f"Profiles for {agg_val} segments")
        st.dataframe(profile.style.background_gradient(cmap='Greens').format("{:.1f}"), height=300)


