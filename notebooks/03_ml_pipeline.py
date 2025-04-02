#!/usr/bin/env python
# coding: utf-8

# # Online Retail ML Pipeline with MLflow
# 
# This notebook implements a machine learning pipeline for the Online Retail dataset using MLflow.
# It performs the following steps:
# 1. Load the transformed RFM data
# 2. Prepare features for clustering
# 3. Train and evaluate clustering models
# 4. Track experiments with MLflow
# 5. Save the best model

import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import mlflow
import mlflow.sklearn
import joblib

# Set paths
RFM_DATA_PATH = "/home/ubuntu/big_data_project/data/curated/rfm_data.csv"
MODELS_DIR = "/home/ubuntu/big_data_project/models"
IMAGES_DIR = "/home/ubuntu/big_data_project/images"

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# Set MLflow experiment
mlflow.set_experiment("Online Retail Customer Segmentation")

print("Starting Online Retail ML Pipeline...")

# Function to load RFM data
def load_rfm_data(file_path):
    """
    Load RFM data from CSV file
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

# Function to prepare features for clustering
def prepare_features(df):
    """
    Prepare features for clustering
    """
    print("\n=== Preparing Features ===")
    
    # Select RFM features
    features_df = df[['Recency', 'Frequency', 'Monetary']].copy()
    
    # Log transform for Monetary (which is usually skewed)
    features_df['Monetary'] = np.log1p(features_df['Monetary'])
    
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_df)
    
    # Create a DataFrame with scaled features
    scaled_df = pd.DataFrame(scaled_features, columns=features_df.columns)
    
    print("Features prepared and standardized.")
    print("\nScaled features summary:")
    print(scaled_df.describe())
    
    return scaled_df, scaler

# Function to find optimal number of clusters
def find_optimal_clusters(scaled_features, max_clusters=10):
    """
    Find optimal number of clusters using silhouette score
    """
    print("\n=== Finding Optimal Number of Clusters ===")
    
    silhouette_scores = []
    inertia_values = []
    
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_features)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(scaled_features, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        
        # Calculate inertia (within-cluster sum of squares)
        inertia_values.append(kmeans.inertia_)
        
        print(f"For n_clusters = {n_clusters}, the silhouette score is {silhouette_avg:.3f}")
    
    # Plot silhouette scores
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.title('Silhouette Score Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    
    # Plot elbow method
    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_clusters + 1), inertia_values, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/optimal_clusters.png")
    print(f"Saved optimal clusters plot to {IMAGES_DIR}/optimal_clusters.png")
    
    # Find optimal number of clusters based on silhouette score
    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
    print(f"\nOptimal number of clusters based on silhouette score: {optimal_clusters}")
    
    return optimal_clusters, silhouette_scores, inertia_values

# Function to train K-means clustering model
def train_kmeans_model(scaled_features, n_clusters, with_mlflow=True):
    """
    Train K-means clustering model and track with MLflow
    """
    print(f"\n=== Training K-means Model with {n_clusters} Clusters ===")
    
    if with_mlflow:
        with mlflow.start_run(run_name=f"kmeans_{n_clusters}_clusters"):
            # Log parameters
            mlflow.log_param("n_clusters", n_clusters)
            mlflow.log_param("algorithm", "k-means")
            mlflow.log_param("random_state", 42)
            
            # Train model
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(scaled_features)
            
            # Evaluate model
            cluster_labels = kmeans.labels_
            silhouette_avg = silhouette_score(scaled_features, cluster_labels)
            
            # Log metrics
            mlflow.log_metric("silhouette_score", silhouette_avg)
            mlflow.log_metric("inertia", kmeans.inertia_)
            
            # Log model
            mlflow.sklearn.log_model(kmeans, "kmeans_model")
            
            print(f"Model trained with silhouette score: {silhouette_avg:.3f}")
            print(f"Model logged to MLflow with run_id: {mlflow.active_run().info.run_id}")
            
            return kmeans, cluster_labels
    else:
        # Train model without MLflow tracking
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(scaled_features)
        cluster_labels = kmeans.labels_
        silhouette_avg = silhouette_score(scaled_features, cluster_labels)
        print(f"Model trained with silhouette score: {silhouette_avg:.3f}")
        
        return kmeans, cluster_labels

# Function to visualize clusters
def visualize_clusters(scaled_features, cluster_labels, n_clusters):
    """
    Visualize clusters using PCA for dimensionality reduction
    """
    print("\n=== Visualizing Clusters ===")
    
    # Apply PCA to reduce to 2 dimensions for visualization
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_features)
    
    # Create DataFrame with principal components and cluster labels
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = cluster_labels
    
    # Plot clusters
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis', s=50, alpha=0.7)
    plt.title(f'Customer Segments - {n_clusters} Clusters')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/cluster_visualization.png")
    print(f"Saved cluster visualization to {IMAGES_DIR}/cluster_visualization.png")
    
    return pca

# Function to analyze clusters
def analyze_clusters(original_df, cluster_labels, n_clusters):
    """
    Analyze characteristics of each cluster
    """
    print("\n=== Analyzing Clusters ===")
    
    # Add cluster labels to original data
    df_with_clusters = original_df.copy()
    df_with_clusters['Cluster'] = cluster_labels
    
    # Analyze cluster characteristics
    cluster_analysis = df_with_clusters.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'CustomerID': 'count'
    }).reset_index()
    
    cluster_analysis.columns = ['Cluster', 'Avg_Recency', 'Avg_Frequency', 'Avg_Monetary', 'Customer_Count']
    cluster_analysis['Cluster_Size_Percentage'] = (cluster_analysis['Customer_Count'] / cluster_analysis['Customer_Count'].sum()) * 100
    
    print("\nCluster Analysis:")
    print(cluster_analysis)
    
    # Visualize cluster characteristics
    plt.figure(figsize=(15, 10))
    
    # Plot average recency by cluster
    plt.subplot(2, 2, 1)
    sns.barplot(x='Cluster', y='Avg_Recency', data=cluster_analysis)
    plt.title('Average Recency by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Average Recency (days)')
    
    # Plot average frequency by cluster
    plt.subplot(2, 2, 2)
    sns.barplot(x='Cluster', y='Avg_Frequency', data=cluster_analysis)
    plt.title('Average Frequency by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Average Frequency')
    
    # Plot average monetary by cluster
    plt.subplot(2, 2, 3)
    sns.barplot(x='Cluster', y='Avg_Monetary', data=cluster_analysis)
    plt.title('Average Monetary Value by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Average Monetary Value')
    
    # Plot cluster size
    plt.subplot(2, 2, 4)
    sns.barplot(x='Cluster', y='Cluster_Size_Percentage', data=cluster_analysis)
    plt.title('Cluster Size Distribution')
    plt.xlabel('Cluster')
    plt.ylabel('Percentage of Customers')
    
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/cluster_analysis.png")
    print(f"Saved cluster analysis to {IMAGES_DIR}/cluster_analysis.png")
    
    # Save cluster analysis to CSV
    cluster_analysis.to_csv(f"/home/ubuntu/big_data_project/data/curated/cluster_analysis.csv", index=False)
    print(f"Saved cluster analysis to /home/ubuntu/big_data_project/data/curated/cluster_analysis.csv")
    
    return cluster_analysis

# Function to save model and artifacts
def save_model_artifacts(kmeans_model, scaler, pca, n_clusters):
    """
    Save model and related artifacts
    """
    print("\n=== Saving Model Artifacts ===")
    
    # Save K-means model
    model_path = f"{MODELS_DIR}/kmeans_model_{n_clusters}_clusters.pkl"
    joblib.dump(kmeans_model, model_path)
    print(f"K-means model saved to {model_path}")
    
    # Save scaler
    scaler_path = f"{MODELS_DIR}/scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    # Save PCA
    pca_path = f"{MODELS_DIR}/pca.pkl"
    joblib.dump(pca, pca_path)
    print(f"PCA model saved to {pca_path}")
    
    # Create a simple model registry file
    registry_info = {
        'model_name': 'customer_segmentation',
        'model_version': '1.0',
        'model_path': model_path,
        'scaler_path': scaler_path,
        'pca_path': pca_path,
        'n_clusters': n_clusters,
        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    registry_df = pd.DataFrame([registry_info])
    registry_df.to_csv(f"{MODELS_DIR}/model_registry.csv", index=False)
    print(f"Model registry updated at {MODELS_DIR}/model_registry.csv")

# Function to demonstrate MLflow in Databricks (for documentation)
def mlflow_databricks_example():
    """
    Example of how MLflow would be used in Databricks
    This is for documentation purposes only and won't be executed
    """
    print("\n=== MLflow in Databricks Example (Documentation Only) ===")
    print("The following code demonstrates how MLflow would be used in a Databricks environment:")
    
    mlflow_code = """
    # In Databricks, MLflow is automatically configured
    import mlflow
    import mlflow.sklearn
    from mlflow.models.signature import infer_signature
    
    # Load data from Delta Lake
    rfm_data = spark.read.format("delta").load("/mnt/data/curated/rfm_data")
    
    # Convert to pandas for scikit-learn
    pdf = rfm_data.toPandas()
    features = pdf[['Recency', 'Frequency', 'Monetary']]
    
    # Standardize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Start MLflow experiment
    with mlflow.start_run(run_name="kmeans_clustering"):
        # Log parameters
        mlflow.log_param("n_clusters", 5)
        mlflow.log_param("algorithm", "k-means")
        
        # Train model
        kmeans = KMeans(n_clusters=5, random_state=42)
        kmeans.fit(scaled_features)
        
        # Log metrics
        silhouette_avg = silhouette_score(scaled_features, kmeans.labels_)
        mlflow.log_metric("silhouette_score", silhouette_avg)
        mlflow.log_metric("inertia", kmeans.inertia_)
        
        # Log model with signature
        signature = infer_signature(scaled_features, kmeans.labels_)
        mlflow.sklearn.log_model(kmeans, "kmeans_model", signature=signature)
        
        # Log artifacts
        mlflow.log_artifact("cluster_visualization.png")
        
    # Register model in MLflow Model Registry
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/kmeans_model"
    registered_model = mlflow.register_model(model_uri, "customer_segmentation")
    
    # Deploy model to serving endpoint
    from mlflow.deployments import get_deploy_client
    client = get_deploy_client("databricks")
    deployment = client.create_endpoint(
        name="customer-segmentation-endpoint",
        config={"registered_model_name": "customer_segmentation"}
    )
    """
    
    print(mlflow_code)

# Main execution function
def main():
    """
    Main execution function
    """
    # Load RFM data
    rfm_df = load_rfm_data(RFM_DATA_PATH)
    
    # Prepare features
    scaled_features, scaler = prepare_features(rfm_df)
    
    # Find optimal number of clusters
    optimal_clusters, silhouette_scores, inertia_values = find_optimal_clusters(scaled_features)
    
    # Train K-means model with optimal number of clusters
    kmeans_model, cluster_labels = train_kmeans_model(scaled_features, optimal_clusters)
    
    # Visualize clusters
    pca_model = visualize_clusters(scaled_features, cluster_labels, optimal_clusters)
    
    # Analyze clusters
    cluster_analysis = analyze_clusters(rfm_df, cluster_labels, optimal_clusters)
    
    # Save model and artifacts
    save_model_artifacts(kmeans_model, scaler, pca_model, optimal_clusters)
    
    # Show MLflow in Databricks example (for documentation)
    mlflow_databricks_example()
    
    print("\nML pipeline completed successfully!")
    print(f"Model and artifacts saved to: {MODELS_DIR}")
    print(f"Visualizations saved to: {IMAGES_DIR}")

# Execute the pipeline
if __name__ == "__main__":
    main()
