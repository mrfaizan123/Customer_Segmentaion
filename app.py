import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("🛍️ Customer Segmentation using K-Means")

# -------------------------
# Upload dataset
# -------------------------
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    customer_data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(customer_data.head())

    # -------------------------
    # Select features
    # -------------------------
    st.write("### Select Features for Clustering")
    columns = customer_data.columns.tolist()
    selected_features = st.multiselect("Choose 2 features for clustering:", columns, default=[columns[3], columns[4]])

    if len(selected_features) == 2:
        X = customer_data[selected_features].values

        # -------------------------
        # Elbow Method
        # -------------------------
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)

        fig, ax = plt.subplots()
        sns.set()
        ax.plot(range(1, 11), wcss, marker='o')
        ax.set_title("The Elbow Point Graph")
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("WCSS")
        st.pyplot(fig)

        # -------------------------
        # Choose K & Run KMeans
        # -------------------------
        k = st.slider("Select number of clusters (K)", 2, 10, 5)
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        Y = kmeans.fit_predict(X)

        customer_data["Cluster"] = Y
        st.write("### Clustered Data")
        st.dataframe(customer_data.head())

        # -------------------------
        # Plot Clusters
        # -------------------------
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        colors = ["green", "red", "yellow", "violet", "blue", "orange", "pink", "cyan", "brown", "purple"]

        for cluster in range(k):
            ax2.scatter(
                X[Y == cluster, 0], X[Y == cluster, 1],
                s=50, c=colors[cluster], label=f'Cluster {cluster+1}'
            )

        # plot centroids
        ax2.scatter(
            kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=200, c="black", marker="X", label="Centroids"
        )

        ax2.set_title("Customer Segments")
        ax2.set_xlabel(selected_features[0])
        ax2.set_ylabel(selected_features[1])
        ax2.legend()
        st.pyplot(fig2)

        # -------------------------
        # Predict new point
        # -------------------------
        st.write("### Predict Cluster for New Customer")
        val1 = st.number_input(f"Enter {selected_features[0]}", min_value=0.0, step=1.0)
        val2 = st.number_input(f"Enter {selected_features[1]}", min_value=0.0, step=1.0)

        if st.button("Predict Cluster"):
            cluster_pred = kmeans.predict([[val1, val2]])[0]
            st.success(f"👉 This customer belongs to **Cluster {cluster_pred+1}**")

else:
    st.info("👆 Upload a dataset to get started!")
