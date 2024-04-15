# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 10:07:02 2024

@author: BeButton
"""

#%%  Packages

import random
import pandas as pd
import numpy as np
# from datetime import datetime

import matplotlib.cm as cm
import matplotlib.pyplot as plt

from hdbscan import HDBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score

#%%  Utils

#  change data labels
def change_lab(y_labels):
    labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', -1:'HH'}
    y_fixed = [labels[num] for num in y_labels]
    return np.array(y_fixed)

#  Silhuoette plot
def silhuoette_plot(x_train, y_labels, title="Silhuoette plot"):
    sample_silhouette_values = silhouette_samples(x_train, y_labels)
    silhouette_avg = silhouette_score(x_train, y_labels)
    n_cluster = len(np.unique(y_labels))

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    y_lower =10
    for i in reversed(range(n_cluster)):
        if 'DBSCAN ' in title:
            ith_cluster_silhouette_values = sample_silhouette_values[y_labels == i - 1]
            lab = change_lab([i - 1])
        else:
            ith_cluster_silhouette_values = sample_silhouette_values[y_labels == i]
            lab = change_lab([i])
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = cm.nipy_spectral(float(i) / n_cluster)
        ax.fill_betweenx(np.arange(y_lower,y_upper),0,ith_cluster_silhouette_values,facecolor=color,edgecolor=color,alpha=0.75)
        
        #label the silhouse plots with their cluster numbers at the middle
        ax.text(-0.05,y_lower + 0.5 * size_cluster_i, lab[0])
        
        #compute the new y_lower for next plot
        y_lower = y_upper +10 
        
    ax.set_title(title)
    ax.set_xlabel("silhouette score")
    ax.set_ylabel("Cluster label")
        
    #the vertical line for average silhouette score of all the values
    ax.axvline(x=silhouette_avg, color="red",linestyle="--")
        
    ax.set_yticks([])
    ax.set_xticks([-0.2,0,0.2,0.4,0.6,0.8,1])
    
    return silhouette_avg

# Cros Tables
def cros_table(y_labels):
    lab = change_lab(y_labels)
    df_check = pd.DataFrame({'predicted': lab, 'original': y[2]})
    ct = pd.crosstab(df_check['predicted'], df_check['original'])
    print(ct)
    return ct

#%%  Data

random.seed(123)
df = pd.read_table('aggregation.csv', sep='\s+', header=None)

X = df[[0,1]]
y = df[[2]]

mean = df.groupby(2).mean()

#%%  First Look

plt.scatter(X[0], X[1])
plt.scatter(mean[0], mean[1], marker='D', s=30, c='red')
plt.show()

#%%  Original data

plt.scatter(X[0], X[1], c=y, alpha=0.8)
plt.colorbar(label='Cluster')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Original Clustering')
plt.show()

#%%  Original Silhouette
print(f'Silhouette Score(orginal data n=7): {silhouette_score(X, y[2])}')

#%%  Kmeans

kmeans = KMeans(n_clusters=7, random_state=10)
y_mean = kmeans.fit_predict(X)        # c = clusterer.labels_
centroids = kmeans.cluster_centers_

plt.scatter(X[0], X[1], c=y_mean, alpha=0.8)
plt.colorbar(label='Cluster') 
plt.scatter(centroids[:, 0], centroids[:, 1], marker='D', s=30, c='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('KMeans Cluster')
plt.show()

#%%
print(f'Silhouette Score(KMeans n=7): {silhouette_score(X, y_mean)}')

#%%
silhuoette_plot(X, y_mean, 'Kmeans silhouette plot')

#%%
cros_table(y_mean)

#%%  Nearest Neighbors - distance parameter

nn = NearestNeighbors(n_neighbors=7)
nbrs = nn.fit(X)
distances, indices = nbrs.kneighbors(X)

# Plotting K-distance Graph
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.figure(figsize=(6,6))
plt.plot(distances)
plt.title('K-distance Graph for "Dart df"',fontsize=20)
plt.xlabel('Data Points sorted by distance',fontsize=14)
plt.ylabel('Epsilon',fontsize=14)
plt.show()

#%%
print(round(max(distances), 2))

#%%  DBScan

db = DBSCAN(eps=1.4, min_samples=7)
y_DBScan = db.fit_predict(X)          # labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(y_DBScan)) - (1 if -1 in y_DBScan else 0)
n_noise_ = list(y_DBScan).count(-1)

df_db = X.copy()
df_db[2] = pd.Series(y_DBScan)
centroids = df_db.groupby(2).mean()

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)

plt.scatter(X[0], X[1], c=y_DBScan)
plt.colorbar(label='Cluster') 
plt.scatter(centroids.iloc[1:, [0]], centroids.iloc[1:, [1]], marker='D', s=30, c='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('DBSCAN Cluster')
plt.show()

#%%
print(f'Silhouette Score(DBSCAN n=7): {silhouette_score(X, y_DBScan)}')

#%%
silhuoette_plot(X, y_DBScan, 'DBSCAN silhouette plot')

#%%
cros_table(y_DBScan)

#%%   HDBScan
hdb = HDBSCAN(min_cluster_size=5,                            # default
                            min_samples=None,                # default
                            cluster_selection_epsilon=1.4,
                            cluster_selection_method='leaf',
                            allow_single_cluster=False,      # default
                            metric='euclidean',              # default
                            algorithm='best',
                            leaf_size=30)
hdb.fit(X)
y_HDBScan = hdb.fit_predict(X)       # labels = hdb.labels_
HDBprob = hdb.probabilities_

df_hdb = X.copy()
df_hdb[2] = pd.Series(y_HDBScan)
centroids = df_db.groupby(2).mean()

n_clusters_ = len(set(y_HDBScan)) - (1 if -1 in y_HDBScan else 0)
n_noise_ = list(y_HDBScan).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)

plt.scatter(X[0], X[1], c=y_HDBScan)
plt.colorbar(label='Cluster') 
plt.scatter(centroids.iloc[1:, [0]], centroids.iloc[1:, [1]], marker='D', s=30, c='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('HDBSCAN Cluster')
plt.show()

#%%
print(f'Silhouette Score(HDBSCAN n=7): {silhouette_score(X, y_HDBScan)}')

#%%
silhuoette_plot(X, y_HDBScan, 'HDBSCAN silhouette plot')

#%%
cros_table(y_HDBScan)

#%%  Gaussian Mixture

best_model = {'gmm': None, 'score': 0, 'y_GMM': []}

for i in range(1,101):
    gmm = GaussianMixture(n_components=7, n_init=10)
    y_GMM = gmm.fit_predict(X)
    score = silhouette_score(X, y_GMM)
    print(f'Silhouette Score(GaussianMixture n=7): {score}')
    if best_model['score'] < score:
        best_model['score'] = score
        best_model['gmm'] = gmm
        best_model['y_GMM'] = y_GMM

#%%
df_gmm = X.copy()
df_gmm[2] = pd.Series(best_model['y_GMM'])
centroids = df_gmm.groupby(2).mean()

plt.scatter(X[0], X[1], c=best_model['y_GMM'])
plt.colorbar(label='Cluster') 
plt.scatter(centroids[0], centroids[1], marker='D', s=30, c='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Gaussian Mixture Cluster')
plt.show()

#%%
print(f'Silhouette Score(Gaussian Mixture n=7): {best_model["score"]}')

#%%
silhuoette_plot(X, best_model['y_GMM'], 'Gaussian Mixture silhouette plot')

#%%
cros_table(best_model['y_GMM'])

#%%
print(f'Silhouette Score(orginal data n=7): {silhouette_score(X, y[2])}')

#%%
