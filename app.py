import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation Pro",
    layout="wide",
    page_icon="🛍️",
    initial_sidebar_state="expanded"
)

# Advanced Custom CSS with modern styling - FIXED FOR LARGE SCREENS
st.markdown("""
<style>
    /* Global Styles */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-size: 1.1rem !important;
    }

    /* Unified Card Style */
    .custom-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 1.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        font-size: 1.1rem;
    }

    .custom-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.3);
    }

    /* Header Styles - LARGER FOR BIG SCREENS */
    .main-header {
        font-size: 4rem !important;
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    .section-header {
        font-size: 2.5rem !important;
        color: #2c3e50;
        margin: 2rem 0 1.5rem 0;
        font-weight: 700;
        border-left: 6px solid #667eea;
        padding-left: 1.5rem;
    }

    /* Metric Cards - LARGER */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px 0 rgba(0,0,0,0.1);
        font-size: 1.1rem;
    }

    .metric-card h2 {
        font-size: 2.5rem !important;
        margin: 0.5rem 0;
    }

    .metric-card h3 {
        font-size: 1.5rem !important;
        margin: 0.5rem 0;
    }

    /* Feature Boxes */
    .feature-box {
        background: rgba(255, 255, 255, 0.9);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        transition: all 0.3s ease;
        font-size: 1.1rem;
    }

    .feature-box:hover {
        background: rgba(255, 255, 255, 1);
        transform: scale(1.02);
    }

    .feature-box h4 {
        font-size: 1.4rem !important;
        color: #2c3e50;
        margin-bottom: 1rem;
    }

    /* Button Styles - LARGER */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 1rem 3rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
        font-size: 1.2rem;
    }

    .stButton button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }

    /* Text Sizes - LARGER */
    .stMarkdown {
        font-size: 1.1rem !important;
    }

    .stNumberInput input {
        font-size: 1.2rem !important;
        padding: 1rem !important;
    }

    .stSelectbox label, .stMultiselect label {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
    }

    .stSlider label {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
    }

    /* File Uploader - LARGER */
    .uploadedFile {
        border: 3px dashed #667eea;
        border-radius: 15px;
        padding: 3rem;
        text-align: center;
        font-size: 1.2rem;
    }

    /* Dataframe Styling */
    .dataframe {
        font-size: 1.1rem !important;
    }

    /* Make all text larger */
    .stApp h1 {
        font-size: 3rem !important;
    }

    .stApp h2 {
        font-size: 2.5rem !important;
    }

    .stApp h3 {
        font-size: 2rem !important;
    }

    .stApp h4 {
        font-size: 1.5rem !important;
    }

    .stApp p, .stApp li, .stApp span {
        font-size: 1.2rem !important;
        line-height: 1.6 !important;
    }

    /* Responsive container */
    .main .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 95%;
    }

    /* Larger icons */
    .icon-large {
        font-size: 4rem !important;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header Section with animated gradient - UPDATED WITH LARGER TEXT
st.markdown("""
<div class="custom-card">
    <h1 class="main-header">🚀 Customer Segmentation Pro</h1>
    <p style='text-align: center; font-size: 1.5rem; color: #555; margin-bottom: 2rem;'>
    Advanced AI-Powered Customer Analytics Platform
    </p>
</div>
""", unsafe_allow_html=True)

# Introduction Section with unified card style - UPDATED WITH LARGER TEXT
st.markdown("""
<div class="custom-card">
    <h2 style='color: #2c3e50; margin-bottom: 2rem; font-size: 2.5rem;'>🎯 What is Customer Segmentation?</h2>
    <div style='display: grid; grid-template-columns: 2fr 1fr; gap: 3rem;'>
        <div>
            <p style='font-size: 1.3rem; line-height: 1.8; color: #555;'>
            <strong>Customer segmentation</strong> is a powerful marketing strategy that divides a company's customers 
            into groups based on shared characteristics. Using <strong>Machine Learning algorithms</strong> like K-Means clustering, 
            we can automatically discover natural patterns in your customer data and create meaningful segments for 
            targeted marketing, personalized experiences, and improved customer retention.
            </p>
            <p style='font-size: 1.3rem; line-height: 1.8; color: #555; margin-top: 1.5rem;'>
            This advanced platform helps businesses unlock hidden insights from their customer data, enabling 
            data-driven decision making and strategic marketing initiatives.
            </p>
        </div>
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   color: white; padding: 2rem; border-radius: 15px; text-align: center;'>
            <h3 style='margin-bottom: 1.5rem; font-size: 1.8rem;'>📈 ROI Boost</h3>
            <p style='font-size: 3rem; font-weight: bold; margin: 0;'>+47%</p>
            <p style='font-size: 1.2rem;'>Average increase in marketing efficiency</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Benefits Section with modern cards - UPDATED WITH LARGER TEXT
st.markdown('<div class="section-header">💡 Key Benefits</div>', unsafe_allow_html=True)

benefits_col1, benefits_col2, benefits_col3 = st.columns(3)

with benefits_col1:
    st.markdown("""
    <div class="feature-box">
        <h4 style='color: #2c3e50; margin-bottom: 1.5rem;'>🎯 Targeted Marketing</h4>
        <p style='color: #555; font-size: 1.2rem;'>Create personalized campaigns for each customer segment, increasing conversion rates and customer engagement.</p>
    </div>
    """, unsafe_allow_html=True)

with benefits_col2:
    st.markdown("""
    <div class="feature-box">
        <h4 style='color: #2c3e50; margin-bottom: 1.5rem;'>💰 Cost Efficiency</h4>
        <p style='color: #555; font-size: 1.2rem;'>Optimize marketing spend by focusing resources on high-value segments and reducing wasted outreach.</p>
    </div>
    """, unsafe_allow_html=True)

with benefits_col3:
    st.markdown("""
    <div class="feature-box">
        <h4 style='color: #2c3e50; margin-bottom: 1.5rem;'>📊 Data-Driven Insights</h4>
        <p style='color: #555; font-size: 1.2rem;'>Uncover hidden patterns in customer behavior and make informed strategic decisions based on data.</p>
    </div>
    """, unsafe_allow_html=True)

# How It Works Section - UPDATED WITH LARGER TEXT
st.markdown("""
<div class="custom-card">
    <h2 style='color: #2c3e50; margin-bottom: 2rem; font-size: 2.5rem;'>🔧 How It Works</h2>
    <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 2rem;'>
        <div style='text-align: center; padding: 2rem; background: rgba(102, 126, 234, 0.1); border-radius: 15px;'>
            <div style='font-size: 3rem; margin-bottom: 1.5rem;'>📁</div>
            <h4 style='color: #2c3e50; font-size: 1.5rem;'>Upload Data</h4>
            <p style='color: #555; font-size: 1.1rem;'>Provide your customer dataset in CSV format</p>
        </div>
        <div style='text-align: center; padding: 2rem; background: rgba(102, 126, 234, 0.1); border-radius: 15px;'>
            <div style='font-size: 3rem; margin-bottom: 1.5rem;'>🎯</div>
            <h4 style='color: #2c3e50; font-size: 1.5rem;'>Select Features</h4>
            <p style='color: #555; font-size: 1.1rem;'>Choose key customer attributes for analysis</p>
        </div>
        <div style='text-align: center; padding: 2rem; background: rgba(102, 126, 234, 0.1); border-radius: 15px;'>
            <div style='font-size: 3rem; margin-bottom: 1.5rem;'>📈</div>
            <h4 style='color: #2c3e50; font-size: 1.5rem;'>Analyze</h4>
            <p style='color: #555; font-size: 1.1rem;'>AI algorithms find optimal customer segments</p>
        </div>
        <div style='text-align: center; padding: 2rem; background: rgba(102, 126, 234, 0.1); border-radius: 15px;'>
            <div style='font-size: 3rem; margin-bottom: 1.5rem;'>🚀</div>
            <h4 style='color: #2c3e50; font-size: 1.5rem;'>Implement</h4>
            <p style='color: #555; font-size: 1.1rem;'>Apply insights to your marketing strategy</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Common Features Section - UPDATED WITH LARGER TEXT
st.markdown('<div class="section-header">📊 Common Segmentation Features</div>', unsafe_allow_html=True)

features_col1, features_col2, features_col3 = st.columns(3)

with features_col1:
    st.markdown("""
    <div class="custom-card">
        <h4 style='color: #2c3e50; margin-bottom: 1.5rem; font-size: 1.5rem;'>💰 Spending Behavior</h4>
        <ul style='color: #555; line-height: 2.0; font-size: 1.2rem;'>
        <li><strong>Annual Income</strong> - Customer earning capacity</li>
        <li><strong>Spending Score</strong> - Purchasing behavior index</li>
        <li><strong>Purchase Frequency</strong> - How often they buy</li>
        <li><strong>Average Order Value</strong> - Typical spend amount</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with features_col2:
    st.markdown("""
    <div class="custom-card">
        <h4 style='color: #2c3e50; margin-bottom: 1.5rem; font-size: 1.5rem;'>👥 Demographic Data</h4>
        <ul style='color: #555; line-height: 2.0; font-size: 1.2rem;'>
        <li><strong>Age Group</strong> - Generational segmentation</li>
        <li><strong>Location</strong> - Geographic patterns</li>
        <li><strong>Occupation</strong> - Professional background</li>
        <li><strong>Education Level</strong> - Academic qualifications</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with features_col3:
    st.markdown("""
    <div class="custom-card">
        <h4 style='color: #2c3e50; margin-bottom: 1.5rem; font-size: 1.5rem;'>📱 Engagement Metrics</h4>
        <ul style='color: #555; line-height: 2.0; font-size: 1.2rem;'>
        <li><strong>Website Visits</strong> - Online engagement</li>
        <li><strong>Email Opens</strong> - Communication responsiveness</li>
        <li><strong>Social Media</strong> - Brand interaction level</li>
        <li><strong>Customer Lifetime Value</strong> - Long-term value</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Divider with style
st.markdown("""
<div style='height: 3px; background: linear-gradient(90deg, transparent, #667eea, transparent); margin: 4rem 0;'></div>
""", unsafe_allow_html=True)

# -------------------------
# MAIN APPLICATION SECTION - UPDATED WITH LARGER TEXT
# -------------------------
st.markdown("""
<div class="custom-card">
    <h2 style='color: #2c3e50; text-align: center; margin-bottom: 2rem; font-size: 2.5rem;'>🚀 Start Your Analysis</h2>
    <p style='text-align: center; color: #555; font-size: 1.4rem;'>
    Upload your customer data below to begin discovering valuable customer segments
    </p>
</div>
""", unsafe_allow_html=True)

# Upload dataset section - UPDATED WITH LARGER TEXT
upload_col1, upload_col2 = st.columns([2, 1])

with upload_col1:
    uploaded_file = st.file_uploader(
        "**📁 Upload Your Customer Dataset (CSV Format)**",
        type=["csv"],
        help="Upload a CSV file with customer data including numerical features like income, age, spending scores, etc."
    )

with upload_col2:
    st.markdown("""
    <div style='text-align: center; padding: 2rem;'>
        <div style='font-size: 4rem; margin-bottom: 1rem;'>📊</div>
        <p style='color: #555; font-size: 1.2rem;'>Supported: CSV files with numerical data</p>
    </div>
    """, unsafe_allow_html=True)

# MAIN APPLICATION LOGIC - FIXED INDENTATION
if uploaded_file is not None:
    try:
        # Load and display data
        customer_data = pd.read_csv(uploaded_file)

        # Data validation
        if len(customer_data) == 0:
            st.error("❌ The uploaded file is empty. Please upload a valid CSV file with data.")
        else:
            # Dataset overview in metric cards
            st.markdown("""
            <div class="custom-card">
                <h3 style='color: #2c3e50; margin-bottom: 2rem; font-size: 2rem;'>📊 Dataset Overview</h3>
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div style='font-size: 2rem;'>👥</div>
                    <h3>{len(customer_data):,}</h3>
                    <p>Total Customers</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div style='font-size: 2rem;'>📋</div>
                    <h3>{len(customer_data.columns)}</h3>
                    <p>Features</p>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                numeric_cols = len(customer_data.select_dtypes(include=['number']).columns)
                st.markdown(f"""
                <div class="metric-card">
                    <div style='font-size: 2rem;'>🔢</div>
                    <h3>{numeric_cols}</h3>
                    <p>Numerical Features</p>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                missing_vals = customer_data.isnull().sum().sum()
                st.markdown(f"""
                <div class="metric-card">
                    <div style='font-size: 2rem;'>✅</div>
                    <h3>{missing_vals}</h3>
                    <p>Missing Values</p>
                </div>
                """, unsafe_allow_html=True)

            # Data preview
            st.markdown("""
            <div class="custom-card">
                <h3 style='color: #2c3e50; margin-bottom: 1rem; font-size: 2rem;'>👀 Data Preview</h3>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(customer_data.head(10), use_container_width=True)

            # Feature selection
            st.markdown("""
            <div class="custom-card">
                <h3 style='color: #2c3e50; margin-bottom: 1rem; font-size: 2rem;'>🎯 Step 1: Select Features for Clustering</h3>
                <p style='color: #555; margin-bottom: 1rem; font-size: 1.2rem;'>
                Choose two numerical features that will be used to segment your customers. 
                <strong>Pro Tip:</strong> Select features that represent different aspects of customer behavior (e.g., Income vs Spending Score).
                </p>
            </div>
            """, unsafe_allow_html=True)

            columns = customer_data.select_dtypes(include=['number']).columns.tolist()
            if len(columns) < 2:
                st.error("❌ Dataset must contain at least 2 numerical columns for clustering.")
            else:
                selected_features = st.multiselect(
                    "**Select two features:**",
                    columns,
                    default=columns[:2] if len(columns) >= 2 else columns,
                    max_selections=2,
                    help="Choose two numerical features for clustering analysis"
                )

                if len(selected_features) == 2:
                    # Check for missing values in selected features
                    if customer_data[selected_features].isnull().any().any():
                        st.warning(
                            "⚠️ Selected features contain missing values. Rows with missing values will be removed.")
                        customer_data_clean = customer_data.dropna(subset=selected_features)
                        X = customer_data_clean[selected_features].values
                    else:
                        customer_data_clean = customer_data
                        X = customer_data_clean[selected_features].values

                    # Check if we have enough data after cleaning
                    if len(X) < 2:
                        st.error(
                            "❌ Not enough data points after removing missing values. Please select different features or upload a cleaner dataset.")
                    else:
                        # Elbow Method
                        st.markdown("""
                        <div class="custom-card">
                            <h3 style='color: #2c3e50; margin-bottom: 1rem; font-size: 2rem;'>📈 Step 2: Find Optimal Number of Clusters</h3>
                            <p style='color: #555; font-size: 1.2rem;'>
                            The <strong>Elbow Method</strong> helps determine the optimal number of clusters by showing the point where 
                            adding more clusters doesn't significantly improve the model. Look for the 'elbow' in the graph below.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                        with st.spinner('🔍 Analyzing optimal clusters...'):
                            wcss = []
                            max_clusters = min(10, len(X))  # Ensure we don't exceed data points
                            for i in range(1, max_clusters + 1):
                                kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
                                kmeans.fit(X)
                                wcss.append(kmeans.inertia_)

                        # Create elbow plot
                        fig, ax = plt.subplots(figsize=(12, 6))
                        sns.set_style("whitegrid")
                        plt.plot(range(1, max_clusters + 1), wcss, marker='o', linewidth=3, markersize=8,
                                 color='#667eea', markerfacecolor='#764ba2', markeredgewidth=2)
                        plt.title("Elbow Method - Optimal Cluster Analysis", fontsize=16, fontweight='bold', pad=20)
                        plt.xlabel("Number of Clusters", fontsize=12)
                        plt.ylabel("Within-Cluster Sum of Squares (WCSS)", fontsize=12)
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig)

                        # Cluster selection and analysis
                        st.markdown("""
                        <div class="custom-card">
                            <h3 style='color: #2c3e50; margin-bottom: 1rem; font-size: 2rem;'>🎨 Step 3: Create Customer Segments</h3>
                        </div>
                        """, unsafe_allow_html=True)

                        k = st.slider("**Select number of customer segments:**", 2, min(10, len(X)), 3,
                                      help="Choose based on the elbow point in the graph above")

                        with st.spinner(f'🚀 Creating {k} customer segments...'):
                            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
                            Y = kmeans.fit_predict(X)
                            customer_data_clean = customer_data_clean.copy()
                            customer_data_clean["Segment"] = Y + 1

                            # Cluster distribution
                            st.markdown("""
                            <div class="custom-card">
                                <h4 style='color: #2c3e50; margin-bottom: 1rem; font-size: 1.8rem;'>📊 Segment Distribution</h4>
                            </div>
                            """, unsafe_allow_html=True)

                            cluster_counts = customer_data_clean["Segment"].value_counts().sort_index()
                            cols = st.columns(k)
                            for i, col in enumerate(cols):
                                with col:
                                    count = cluster_counts.get(i + 1, 0)
                                    percentage = (count / len(customer_data_clean)) * 100
                                    st.markdown(f"""
                                    <div class="metric-card">
                                        <h3>Segment {i + 1}</h3>
                                        <h2>{count}</h2>
                                        <p>{percentage:.1f}% of customers</p>
                                    </div>
                                    """, unsafe_allow_html=True)

                        # Visualization
                        st.markdown("""
                        <div class="custom-card">
                            <h3 style='color: #2c3e50; margin-bottom: 1rem; font-size: 2rem;'>📊 Step 4: Visualize Customer Segments</h3>
                        </div>
                        """, unsafe_allow_html=True)

                        fig2, ax2 = plt.subplots(figsize=(14, 8))
                        colors = plt.cm.viridis(np.linspace(0, 1, k))

                        for cluster in range(k):
                            ax2.scatter(
                                X[Y == cluster, 0], X[Y == cluster, 1],
                                s=80, c=[colors[cluster]], label=f'Segment {cluster + 1}',
                                alpha=0.7, edgecolors='white', linewidth=1
                            )

                        # Centroids
                        ax2.scatter(
                            kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                            s=400, c="red", marker="X", label="Centroids",
                            edgecolors='black', linewidth=2, alpha=0.9
                        )

                        ax2.set_title("Customer Segmentation Analysis", fontsize=18, fontweight='bold', pad=20)
                        ax2.set_xlabel(selected_features[0], fontsize=14)
                        ax2.set_ylabel(selected_features[1], fontsize=14)
                        ax2.legend(fontsize=12)
                        ax2.grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig2)

                        # Prediction Section
                        st.markdown("""
                        <div class="custom-card">
                            <h3 style='color: #2c3e50; margin-bottom: 1rem; font-size: 2rem;'>🔮 Step 5: Predict New Customer Segment</h3>
                            <p style='color: #555; font-size: 1.2rem;'>
                            Enter details for a new customer to see which segment they would belong to:
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                        pred_col1, pred_col2, pred_col3 = st.columns([1, 1, 1])

                        with pred_col1:
                            val1 = st.number_input(f"**{selected_features[0]}**",
                                                   min_value=0.0, step=0.1, value=0.0,
                                                   help=f"Enter value for {selected_features[0]}")
                        with pred_col2:
                            val2 = st.number_input(f"**{selected_features[1]}**",
                                                   min_value=0.0, step=0.1, value=0.0,
                                                   help=f"Enter value for {selected_features[1]}")
                        with pred_col3:
                            st.write("")  # Spacer
                            st.write("")  # Spacer
                            if st.button("🎯 Predict Customer Segment", type="primary", use_container_width=True):
                                if val1 == 0.0 and val2 == 0.0:
                                    st.warning("⚠️ Please enter values greater than 0 for both features.")
                                else:
                                    with st.spinner('Analyzing customer segment...'):
                                        try:
                                            # Ensure the input is in the correct format (2D array of floats)
                                            input_data = np.array([[float(val1), float(val2)]])
                                            cluster_pred = kmeans.predict(input_data)[0]
                                            st.success(f"**🎉 This customer belongs to Segment {cluster_pred + 1}**")

                                            # Segment profile
                                            segment_data = customer_data_clean[
                                                customer_data_clean["Segment"] == cluster_pred + 1]
                                            st.markdown(f"""
                                            <div class="custom-card">
                                                <h4 style='color: #2c3e50; margin-bottom: 1rem; font-size: 1.8rem;'>📋 Segment {cluster_pred + 1} Profile</h4>
                                                <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;'>
                                                    <div style='background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 10px; border-left: 4px solid #667eea;'>
                                                        <strong style='color: #2c3e50; font-size: 1.2rem;'>Avg {selected_features[0]}</strong><br>
                                                        <span style='color: #667eea; font-size: 1.4rem; font-weight: bold;'>{segment_data[selected_features[0]].mean():.2f}</span>
                                                    </div>
                                                    <div style='background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 10px; border-left: 4px solid #667eea;'>
                                                        <strong style='color: #2c3e50; font-size: 1.2rem;'>Avg {selected_features[1]}</strong><br>
                                                        <span style='color: #667eea; font-size: 1.4rem; font-weight: bold;'>{segment_data[selected_features[1]].mean():.2f}</span>
                                                    </div>
                                                </div>
                                            </div>
                                            """, unsafe_allow_html=True)
                                        except Exception as e:
                                            st.error(f"❌ Error in prediction: {str(e)}")
                                            st.info(
                                                "💡 Make sure you're entering numerical values compatible with your dataset.")

                        # Download results
                        st.markdown("""
                        <div class="custom-card">
                            <h3 style='color: #2c3e50; margin-bottom: 1rem; font-size: 2rem;'>💾 Download Results</h3>
                            <p style='color: #555; font-size: 1.2rem;'>
                            Download your segmented customer data for further analysis or integration with other systems.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                        csv = customer_data_clean.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Segmented Customer Data",
                            data=csv,
                            file_name="customer_segments_analysis.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                else:
                    st.info("👆 Please select exactly 2 features to proceed with clustering analysis.")

    except Exception as e:
        st.error(f"❌ Error processing the file: {str(e)}")
        st.info("💡 Please make sure you've uploaded a valid CSV file with proper formatting.")

else:
    # Empty state with attractive design
    st.markdown("""
    <div class="custom-card" style='text-align: center; padding: 5rem;'>
        <div style='font-size: 5rem; margin-bottom: 2rem;'>📊</div>
        <h2 style='color: #2c3e50; margin-bottom: 2rem; font-size: 2.5rem;'>Ready to Discover Customer Insights?</h2>
        <p style='color: #555; font-size: 1.4rem; margin-bottom: 3rem;'>
        Upload your customer data CSV file to begin your segmentation analysis and unlock valuable customer insights.
        </p>
        <div style='background: rgba(102, 126, 234, 0.1); padding: 2.5rem; border-radius: 15px; display: inline-block;'>
            <p style='color: #667eea; font-weight: bold; margin: 0; font-size: 1.3rem;'>📁 Supported Format: CSV files</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='text-align: center; color: #666; margin-top: 4rem; padding: 3rem;'>
    <p style='font-size: 1.1rem;'>
    <strong>Customer Segmentation Pro</strong> • Built with Streamlit & Scikit-Learn • 
    Transform Your Customer Analytics 🚀
    </p>
</div>
""", unsafe_allow_html=True)
