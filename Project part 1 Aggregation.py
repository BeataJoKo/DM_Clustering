# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 10:04:20 2024

@author: Mathi
"""
#%%
import pandas as pd
import random as rd

from sklearn.metrics import silhouette_samples, silhouette_score

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
rd.seed(123)

df = pd.read_csv("Aggregation.csv")

x = df["x"]
y = df["y"]
clus = df["cluster"]
plt.scatter(x, y, c=clus, cmap='viridis')  
plt.colorbar(label='Cluster') 
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot with Cluster from the data file')
plt.show()



data = list(zip(x, y))
inertias = []
print("Data:", data)
#%%
# Kmeans

kmeans = KMeans(n_clusters=7,n_init=10)
kmeans.fit(data)

plt.scatter(x, y, c=kmeans.labels_)
plt.title("Clusters made with kmeans")
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
print(inertias)





#%%
# Using GaussianMixture
# Clustering with selected number of clusters
num_clusters = 7  
gmm = GaussianMixture(n_components=num_clusters, n_init=10)
gmm.fit(data)

# Plotting clusters
plt.scatter(x, y, c=gmm.predict(data))
plt.title("Clusters made with Gaussian Mixture Model")
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Silhouette Scores
silhouette_GMM_score_list = []
for i in range(2, 11):
    gmm = GaussianMixture(n_components=i, n_init=10)
    gmm.fit(data)
    labels = gmm.predict(data)
    result = silhouette_score(data, labels)
    print("The average Silhouette score for each cluster using GMM with", i, "clusters is", result)
    silhouette_GMM_score_list.append(result)


#%%
#Trying to find the best GMM plot with 7 clusters
best_model = None
best_score = -1  # Initialize with a value lower than any possible silhouette score

for _ in range(50):
    # Clustering with Gaussian Mixture Model
    num_clusters = 7 
    gmm = GaussianMixture(n_components=num_clusters, n_init=10)
    gmm.fit(data)
    
    # Predict cluster labels
    labels = gmm.predict(data)
    
    # Compute silhouette score
    score = silhouette_score(data, labels)
    print(score)
    # Check if the current model has a higher silhouette score than the best model so far
    if score > best_score:
        best_model = gmm
        best_score = score


plt.scatter(x, y, c=best_model.predict(data))
plt.title("Clusters made with the best Gaussian Mixture Model")
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
# After n iterations, print the silhouette score of the best model
print("The highest silhouette score found for Gaussian Mixture Model:", best_score)