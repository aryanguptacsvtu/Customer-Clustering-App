import streamlit as st
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Customer Clustering App", layout="centered")

st.title("🔍 Customer Segmentation using KMeans + t-SNE")
st.write("Upload your dataset, choose features, and explore customer segments interactively.")

# Sidebar for chart theme selection
with st.sidebar:
    st.header("🖌️ Plot Settings")
    theme = st.selectbox("Select Seaborn Theme", ['darkgrid', 'whitegrid', 'dark', 'white', 'ticks'])
    sb.set_style(theme)

# Upload section
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Column selection
    selected_columns = st.multiselect("Select features for clustering", df.columns, default=df.columns)
    df = df[selected_columns]

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Encode categorical features
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = LabelEncoder().fit_transform(df[col])

    # Scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    # Choose K for clustering
    k = st.slider("Choose number of clusters (k)", 2, 10, 4)

    # KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=0)
    segments = kmeans.fit_predict(scaled_data)

    # Evaluation metrics
    st.subheader("📈 Clustering Evaluation")
    st.write(f"Inertia: {kmeans.inertia_:.2f}")
    st.write(f"Silhouette Score: {silhouette_score(scaled_data, segments):.2f}")

    # t-SNE computation (with progress feedback)
    @st.cache_data
    def run_tsne(data):
        tsne = TSNE(n_components=2, perplexity=30, random_state=0)
        return tsne.fit_transform(data)

    st.subheader("🧠 t-SNE Cluster Visualization")
    with st.spinner("Computing t-SNE... please wait ⏳"):
        tsne_data = run_tsne(scaled_data)

    df_tsne = pd.DataFrame({'x': tsne_data[:, 0], 'y': tsne_data[:, 1], 'segment': segments})

    # Plotting
    fig, ax = plt.subplots()
    sb.scatterplot(x='x', y='y', hue='segment', data=df_tsne, palette='Set2', ax=ax)
    st.pyplot(fig)

    # Add cluster label to original data
    df['Cluster'] = segments

    # Cluster summary
    st.subheader("📊 Cluster Summary (mean values)")
    st.dataframe(df.groupby('Cluster').mean().round(2))

    # Download button
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Clustered Data", csv, "clustered_data.csv", "text/csv")
