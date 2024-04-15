# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 10:07:02 2024

@author: BeButton
"""
import pandas as pddddd
#%%

import random
import pandas as pd
import numpy as np
from datetime import datetime

import matplotlib.cm as cm
import matplotlib.pyplot as plt

from hdbscan import HDBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score

#%%
random.seed(123)
df = pd.read_table('aggregation.csv', sep='\s+', header=None)

X = df[[0,1]]
y = df[[2]]

#%%
plt.scatter(X[0], X[1])
plt.show()

#%%
plt.scatter(X[0], X[1], c=y)
plt.show()

#%%

clusterer = KMeans(n_clusters=7, random_state=10)
y_mean = clusterer.fit_predict(X)
centroids = clusterer.cluster_centers_
plt.scatter(X[0], X[1], c=y_mean, alpha=0.8)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='D', s=30, c='red')
plt.show()

#%%
print(f'Silhouette Score(KMeans n=7): {silhouette_score(X, y_mean)}')

#%%

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

#%%
db = DBSCAN(eps=1.4, min_samples=7)
y_DBScan = db.fit_predict(X)

labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)

plt.scatter(X[0], X[1], c=y_DBScan)
plt.show()

#%%
print(f'Silhouette Score(DBSCAN n=7): {silhouette_score(X, y_DBScan)}')

#%%
hdb = HDBSCAN()
hdb.fit(X)
y_HDBScan = hdb.fit_predict(X)

labels = hdb.labels_
HDBprob = hdb.probabilities_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)

plt.scatter(X[0], X[1], c=labels)
plt.show()

#%%
print(f'Silhouette Score(HDBSCAN n=5): {silhouette_score(X, y_HDBScan)}')

#%%
best_model = {'gmm': None, 'score': 0, 'y_GMM': []}

for i in range(1,101):
    gmm = GaussianMixture(n_components=7)
    y_GMM = gmm.fit_predict(X)
    score = silhouette_score(X, y_GMM)
    print(f'Silhouette Score(GaussianMixture n=7): {score}')
    if best_model['score'] < score:
        best_model['score'] = score
        best_model['gmm'] = gmm
        best_model['y_GMM'] = y_GMM

#%%
plt.scatter(X[0], X[1], c=best_model['y_GMM'])
plt.show()
print(f'Silhouette Score(GaussianMixture n=7): {best_model["score"]}')

#%%
sample_silhouette_values = silhouette_samples(X, y_mean)
silhouette_avg = silhouette_score(X, y_mean)
n_cluster = len(np.unique(y_mean))

#%%
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
y_lower =10
for i in range(n_cluster):
    ith_cluster_silhouette_values = sample_silhouette_values[y_mean == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    
    color = cm.nipy_spectral(float(i) / n_cluster)
    ax.fill_betweenx(np.arange(y_lower,y_upper),0,ith_cluster_silhouette_values,facecolor=color,edgecolor=color,alpha=0.75)
    
    #label the silhouse plots with their cluster numbers at the middle
    ax.text(-0.05,y_lower + 0.5 * size_cluster_i,str(i))
    
    #compute the new y_lower for next plot
    y_lower = y_upper +10 
    
ax.set_title("Silhuoette plot")
ax.set_xlabel("silhouette score")
ax.set_ylabel("Cluster label")
    
#the vertical line for average silhouette score of all the values
ax.axvline(x=silhouette_avg, color="red",linestyle="--")
    
ax.set_yticks([])
ax.set_xticks([-0.2,0,0.2,0.4,0.6,0.8,1])

#%%
df_check = pd.DataFrame({'predicted': y_mean, 'original': y[2]})
ct = pd.crosstab(df_check['predicted'], df_check['original'])
print(ct)


#%%
