# -PRODIGY_ML_02-
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Create a synthetic customer purchase history dataset (you should replace this with your own data)
data = {
    'CustomerID': range(1, 11),
    'PurchaseFrequency': [5, 2, 8, 1, 9, 7, 3, 6, 4, 10],
    'TotalSpending': [500, 150, 800, 100, 950, 750, 200, 600, 350, 1000]
}

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Select the features for clustering (e.g., PurchaseFrequency and TotalSpending)
X = df[['PurchaseFrequency', 'TotalSpending']]

# Standardize the features (important for K-means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Choose the number of clusters (K) using the Elbow method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow curve to help determine the optimal number of clusters
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.grid(True)
plt.show()

# Based on the Elbow method, let's choose K=3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Add cluster labels to the DataFrame
df['Cluster'] = kmeans.labels_

# Visualize the clusters using PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame for PCA results
pca_df = pd.DataFrame(data=X_pca, columns=['PCA1', 'PCA2'])
pca_df['Cluster'] = kmeans.labels_

# Plot the clusters
plt.figure(figsize=(10, 6))
plt.scatter(pca_df['PCA1'], pca_df['PCA2'], c=pca_df['Cluster'], cmap='viridis')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('K-means Clustering of Customers')
plt.colorbar(label='Cluster')
plt.show()

# Print cluster centers
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
print("Cluster Centers:")
for i, center in enumerate(cluster_centers):
    print(f"Cluster {i + 1}: PurchaseFrequency = {center[0]:.2f}, TotalSpending = ${center[1]:,.2f}")

