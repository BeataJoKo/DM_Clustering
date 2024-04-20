# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 20:12:55 2024

@author: Mathi
"""

import pandas as pd



from sklearn.metrics import confusion_matrix, recall_score, precision_score

import pandas as pd
import random as rd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
rd.seed(123)

df = pd.read_csv("pathbased.csv")

x = df["x"]
y = df["y"]
clus = df["cluster"]
plt.scatter(x, y, c=clus, cmap='viridis')  
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='Cluster 0', markerfacecolor='turquoise', markersize=8),
    plt.Line2D([0], [0], marker='o', color='w', label='Cluster 1', markerfacecolor='yellow', markersize=8),
    plt.Line2D([0], [0], marker='o', color='w', label='Cluster 2', markerfacecolor='purple', markersize=8)
]
plt.legend(handles=legend_elements)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot with Cluster from the data file')
plt.show()



data = list(zip(x, y))
inertias = []
print("Data:", data)
#%%
# Kmeans
best_model_kmeans = None
best_score_kmeans = -1  # Initialize with a value lower than any possible silhouette score

for _ in range(50):
    # Clustering with Kmeans
    cluster = 3
    kmeans = KMeans(n_clusters=cluster,n_init=10)
    kmeans.fit(data)
    

    
    # Compute silhouette score
    score = silhouette_score(data, kmeans.fit_predict(data))
    print(score)
    # Check if the current model has a higher silhouette score than the best model so far
    if score > best_score_kmeans:
        best_model_kmeans = kmeans
        best_score_kmeans = score
        clusters_kmeans = 3
        labels = kmeans.labels_
        df['cluster_Kmeans']=labels

            

# Plotting clusters
plt.scatter(x, y, c=best_model_kmeans.predict(data))
plt.title("Clusters made with the best kmeans Model")
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='Cluster 0', markerfacecolor='turquoise', markersize=8),
    plt.Line2D([0], [0], marker='o', color='w', label='Cluster 1', markerfacecolor='yellow', markersize=8),
    plt.Line2D([0], [0], marker='o', color='w', label='Cluster 2', markerfacecolor='purple', markersize=8)
]
plt.legend(handles=legend_elements)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
print("The highest silhouette score found for KMeans model:", best_score_kmeans)

#%%
df['cluster_Kmeans'] = df['cluster_Kmeans'].replace({0: 2, 1: 3, 2: 1})
df_check = pd.DataFrame({'Predicted':df['cluster_Kmeans'],"Original":df["cluster"]})
ct_kmeans = pd.crosstab(df_check['Predicted'], df_check['Original'])
print(ct_kmeans)


#%%

best_model_GMM = None
best_score_GMM = -1  # Initialize with a value lower than any possible silhouette score

for _ in range(50):
    # Clustering with Gaussian Mixture Model
    num_clusters = 3
    gmm = GaussianMixture(n_components=num_clusters, n_init=10)
    gmm.fit(data)
    
    # Predict cluster labels
    labels = gmm.predict(data)
    
    # Compute silhouette score
    score = silhouette_score(data, labels)
    print(score)
    # Check if the current model has a higher silhouette score than the best model so far

    if score > best_score_GMM:
        best_model_GMM = gmm
        best_score_GMM = score
        df['cluster_GMM']=labels


plt.scatter(x, y, c=best_model_GMM.predict(data))
plt.title("Clusters made with the best Gaussian Mixture Model")
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='Cluster 0', markerfacecolor='turquoise', markersize=8),
    plt.Line2D([0], [0], marker='o', color='w', label='Cluster 1', markerfacecolor='yellow', markersize=8),
    plt.Line2D([0], [0], marker='o', color='w', label='Cluster 2', markerfacecolor='purple', markersize=8)
]
plt.legend(handles=legend_elements)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
# After n iterations, print the silhouette score of the best model
print("The highest silhouette score found for Gaussian Mixture Model:", best_score_GMM)

#%%
df['cluster_GMM'] = df['cluster_GMM'].replace({0: 3, 1: 2, 2:1 })
df_check = pd.DataFrame({'Predicted':df['cluster_GMM'],"Original":df["cluster"]})
ct_gmm = pd.crosstab(df_check['Predicted'], df_check['Original'])
print(ct_gmm)

#%%


# Clustering with DBSCAN Model we choose the eps_size 2.24 as we have found that to be the most reliable value

eps_size = 2.24
Dbscan_ = DBSCAN(eps=eps_size,min_samples=10).fit(data)


# Predict cluster labels
labels = Dbscan_.labels_
df['cluster_DBSCAN']=Dbscan_.labels_
plt.scatter(x, y, c=labels)
plt.title("Clusters made with the best DBSCAN  Model with an eps size of 2.24")
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='Cluster 0', markerfacecolor='turquoise', markersize=8),
    plt.Line2D([0], [0], marker='o', color='w', label='Cluster 1', markerfacecolor='yellow', markersize=8),
    plt.Line2D([0], [0], marker='o', color='w', label='Cluster 2', markerfacecolor='purple', markersize=8)
]
plt.legend(handles=legend_elements)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

#%%

df['cluster_DBSCAN'] = df['cluster_DBSCAN'].replace({-1: 1, 0: 3, 1: 2})
df_check = pd.DataFrame({'Predicted':df['cluster_DBSCAN'],"Original":df["cluster"]})
ct_dbscan = pd.crosstab(df_check['Predicted'], df_check['Original'])
print(ct_dbscan)



#%%

DBSCAN_precision_score = precision_score(df['cluster'],df['cluster_DBSCAN'],average = "macro")
DBSCAN_recall_score =recall_score(df['cluster'],df['cluster_DBSCAN'],average = "macro")

KMEANS_precision_score = precision_score(df['cluster'],df['cluster_Kmeans'],average = "macro")
KMEANS_recall_score =recall_score(df['cluster'],df['cluster_Kmeans'],average = "macro")

GMM_precision_score = precision_score(df['cluster'],df['cluster_GMM'],average = "macro")
GMM_recall_score =recall_score(df['cluster'],df['cluster_GMM'],average = "macro")


df_precision_Recall = {
    "DBSCAN": [DBSCAN_precision_score, DBSCAN_recall_score],
    "KMEANS": [KMEANS_precision_score, KMEANS_recall_score],
    "GMM": [GMM_precision_score, GMM_recall_score]
}
index = ["Precision", "Recall"]
result_df = pd.DataFrame(df_precision_Recall, index=index)

# Display the DataFrame
print(result_df)