import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from data_processing import load_data, clean_data, scale_data

def perform_kmeans_clustering(df, columns_to_cluster, num_clusters=3):
    """Perform K-Means clustering on the data."""
    
    # Scale the data (important for clustering)
    df_scaled = scale_data(df, columns_to_cluster)
    
    # Fit the KMeans model
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(df[columns_to_cluster])
    
    # Calculate silhouette score for model evaluation
    score = silhouette_score(df[columns_to_cluster], df['cluster'])
    print(f"Silhouette Score: {score}")
    
    return df, kmeans

def plot_clusters(df, columns_to_cluster, kmeans_model):
    """Plot the clusters."""
    plt.figure(figsize=(10, 6))
    plt.scatter(df[columns_to_cluster[0]], df[columns_to_cluster[1]], c=df['cluster'], cmap='viridis', s=50)
    plt.title(f'Clusters of Players (K={len(kmeans_model.cluster_centers_)} clusters)')
    plt.xlabel(columns_to_cluster[0])
    plt.ylabel(columns_to_cluster[1])
    plt.show()

if __name__ == "__main__":
    file_path = "/Users/jhansi/Documents/FootballPlayerPositionPrediction/data/player_stats.csv"
    
    # Load and clean the data
    df = load_data(file_path)
    df = clean_data(df)
    
    # Define columns to cluster
    columns_to_cluster = ['age', 'height', 'weight', 'ballcontrol', 'dribbling', 'aggression']
    
    # Perform KMeans clustering
    df, kmeans_model = perform_kmeans_clustering(df, columns_to_cluster, num_clusters=4)
    
    # Plot the clusters (select two columns for easy visualization)
    plot_clusters(df, columns_to_cluster[:2], kmeans_model)
    
    # Print the resulting clusters' centroids
    print("Cluster Centers:")
    print(kmeans_model.cluster_centers_)
    
    # Show the cluster distribution
    print("\nCluster Distribution:")
    print(df['cluster'].value_counts())
