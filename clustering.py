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
from sklearn.metrics import confusion_matrix, recall_score, precision_score

#%%  Utils

#  change data labels
labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', -1:'HH'}
def change_lab(y_labels, lab=labels):
    y_fixed = [lab[num] for num in y_labels]
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
        if -1 in np.unique(y_labels):
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

# Cros Tab
def cros_table(y_labels, y_data):
    lab = change_lab(y_labels)
    df_check = pd.DataFrame({'predicted': lab, 'original': y_data[2]})
    ct = pd.crosstab(df_check['predicted'], df_check['original'])
    print(ct)
    return ct

#%%  Data

random.seed(123)
df = pd.read_table('aggregation.csv', sep='\s+', header=None)
df2 = pd.read_table('path.csv', sep='\s+', header=None)
df3 = pd.read_csv("pathbased.csv")
df3.columns = [0, 1, 2]

#%%

X = df[[0,1]]
y = df[[2]]

X2 = df2[[0,1]]
y2 = df2[[2]]

X3 = df3[[0,1]]
y3 = df3[[2]]

mean = df.groupby(2).mean()
mean2 = df2.groupby(2).mean()
mean3 = df3.groupby(2).mean()

#%%  First Look

plt.scatter(X[0], X[1])
plt.scatter(mean[0], mean[1], marker='D', s=30, c='red')
plt.show()

plt.scatter(X2[0], X2[1])
plt.scatter(mean2[0], mean2[1], marker='D', s=30, c='red')
plt.show()

plt.scatter(X3[0], X3[1])
plt.scatter(mean3[0], mean3[1], marker='D', s=30, c='red')
plt.show()

#%%  Original data

plt.scatter(X[0], X[1], c=y, alpha=0.8)
plt.colorbar(label='Cluster')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Original Clustering')
plt.show()

plt.scatter(X2[0], X2[1], c=y2, alpha=0.8)
plt.colorbar(label='Cluster')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Original Clustering')
plt.show()

plt.scatter(X3[0], X3[1], c=y3, alpha=0.8)
plt.colorbar(label='Cluster')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Original Clustering')
plt.show()

#%%
print(f'Silhouette Score(orginal data n=7): {silhouette_score(X, y[2])}')
print(f'Silhouette Score(orginal data n=3): {silhouette_score(X2, y2[2])}')
print(f'Silhouette Score(orginal data n=3): {silhouette_score(X3, y3[2])}')

#%%  Kmeans

kmeans = KMeans(n_clusters=7, random_state=10)
y_mean = kmeans.fit_predict(X)        # c = clusterer.labels_
centroids = kmeans.cluster_centers_

kmeans2 = KMeans(n_clusters=3, random_state=10)
y_mean2 = kmeans2.fit_predict(X2)        # c = clusterer.labels_
centroids2 = kmeans2.cluster_centers_

kmeans3 = KMeans(n_clusters=3, random_state=10)
y_mean3 = kmeans3.fit_predict(X3)        # c = clusterer.labels_
centroids3 = kmeans3.cluster_centers_

#%%

plt.scatter(X[0], X[1], c=y_mean, alpha=0.8)
plt.colorbar(label='Cluster') 
plt.scatter(centroids[:, 0], centroids[:, 1], marker='D', s=30, c='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('KMeans Cluster')
plt.show()

plt.scatter(X2[0], X2[1], c=y_mean2, alpha=0.8)
plt.colorbar(label='Cluster') 
plt.scatter(centroids2[:, 0], centroids2[:, 1], marker='D', s=30, c='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('KMeans Cluster')
plt.show()

plt.scatter(X3[0], X3[1], c=y_mean3, alpha=0.8)
plt.colorbar(label='Cluster') 
plt.scatter(centroids3[:, 0], centroids3[:, 1], marker='D', s=30, c='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('KMeans Cluster')
plt.show()

#%%
print(f'Silhouette Score(KMeans n=7): {silhouette_score(X, y_mean)}')
print(f'Silhouette Score(KMeans n=3): {silhouette_score(X2, y_mean2)}')
print(f'Silhouette Score(KMeans n=3): {silhouette_score(X3, y_mean3)}')

#%%
silhuoette_plot(X, y_mean, 'Kmeans silhouette plot')
silhuoette_plot(X2, y_mean2, 'Kmeans silhouette plot')
silhuoette_plot(X3, y_mean3, 'Kmeans silhouette plot')

#%%
cros_table(y_mean, y)
cros_table(y_mean2, y2)
cros_table(y_mean3, y3)

#%% renaming

df['kMeans'] = change_lab(y_mean, {0: 4, 1: 2, 2: 6, 3: 3, 4: 4, 5: 1, 6:5})
df2['kMeans'] = change_lab(y_mean2, {0: 3, 1: 1, 2: 2})
df3['kMeans'] = change_lab(y_mean3, {0: 2, 1: 3, 2: 1})

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

db2 = DBSCAN(eps=3.24, min_samples=5)
y_DBScan2 = db2.fit_predict(X2)          # labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_2 = len(set(y_DBScan2)) - (1 if -1 in y_DBScan2 else 0)
n_noise_2 = list(y_DBScan2).count(-1)

df_db2 = X2.copy()
df_db2[2] = pd.Series(y_DBScan2)
centroids2 = df_db2.groupby(2).mean()

print("Estimated number of clusters: %d" % n_clusters_2)
print("Estimated number of noise points: %d" % n_noise_2)

db3 = DBSCAN(eps=2.24, min_samples=10)
y_DBScan3 = db3.fit_predict(X3)          # labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_3 = len(set(y_DBScan3)) - (1 if -1 in y_DBScan3 else 0)
n_noise_3 = list(y_DBScan3).count(-1)

df_db3 = X3.copy()
df_db3[2] = pd.Series(y_DBScan3)
centroids3 = df_db3.groupby(2).mean()

print("Estimated number of clusters: %d" % n_clusters_3)
print("Estimated number of noise points: %d" % n_noise_3)

#%%

plt.scatter(X[0], X[1], c=y_DBScan)
plt.colorbar(label='Cluster') 
plt.scatter(centroids.iloc[1:, [0]], centroids.iloc[1:, [1]], marker='D', s=30, c='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('DBSCAN Cluster')
plt.show()

plt.scatter(X2[0], X2[1], c=y_DBScan2)
plt.colorbar(label='Cluster') 
plt.scatter(centroids2.iloc[:, [0]], centroids2.iloc[:, [1]], marker='D', s=30, c='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('DBSCAN Cluster')
plt.show()

plt.scatter(X3[0], X3[1], c=y_DBScan3)
plt.colorbar(label='Cluster') 
plt.scatter(centroids3.iloc[1:, [0]], centroids3.iloc[1:, [1]], marker='D', s=30, c='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('DBSCAN Cluster')
plt.show()

#%%
print(f'Silhouette Score(DBSCAN n=7): {silhouette_score(X, y_DBScan)}')
print(f'Silhouette Score(DBSCAN n=3): {silhouette_score(X2, y_DBScan2)}')
print(f'Silhouette Score(DBSCAN n=2): {silhouette_score(X3, y_DBScan3)}')

#%%
silhuoette_plot(X, y_DBScan, 'DBSCAN silhouette plot')
silhuoette_plot(X2, y_DBScan2, 'DBSCAN silhouette plot')
silhuoette_plot(X3, y_DBScan3, 'DBSCAN silhouette plot')

#%%
cros_table(y_DBScan, y)
cros_table(y_DBScan2, y2)
cros_table(y_DBScan3, y3)

#%%
df['dbScan'] = change_lab(y_DBScan, {0: 2, 1: 7, 2: 4, 3: 3, 4: 6, 5: 1, 6:5, -1:0})
df2['dbScan'] = change_lab(y_DBScan2, {0: 3, 1: 1, 2: 2})
df3['dbScan'] = change_lab(y_DBScan3, {0: 3, 1: 2, -1: 1})

#%%   HDBScan

hdb = HDBSCAN(min_cluster_size=5,                            # default
                            min_samples=None,                # default
                            cluster_selection_epsilon=1.4,
                            cluster_selection_method='leaf',
                            allow_single_cluster=False,      # default
                            metric='euclidean',              # default
                            algorithm='best',
                            leaf_size=30)
y_HDBScan = hdb.fit_predict(X)       # labels = hdb.labels_
HDBprob = hdb.probabilities_

df_hdb = X.copy()
df_hdb[2] = pd.Series(y_HDBScan)
centroids = df_db.groupby(2).mean()

n_clusters_ = len(set(y_HDBScan)) - (1 if -1 in y_HDBScan else 0)
n_noise_ = list(y_HDBScan).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)


hdb2 = HDBSCAN(min_cluster_size=5,                            # default
                            min_samples=None,                # default
                            cluster_selection_epsilon=3.24,
                            cluster_selection_method='leaf',
                            allow_single_cluster=False,      # default
                            metric='euclidean',              # default
                            algorithm='best',
                            leaf_size=20)
y_HDBScan2 = hdb2.fit_predict(X2)       # labels = hdb.labels_

df_hdb2 = X2.copy()
df_hdb2[2] = pd.Series(y_HDBScan2)
centroids2 = df_db2.groupby(2).mean()

n_clusters_2 = len(set(y_HDBScan2)) - (1 if -1 in y_HDBScan2 else 0)
n_noise_2 = list(y_HDBScan2).count(-1)

print("Estimated number of clusters: %d" % n_clusters_2)
print("Estimated number of noise points: %d" % n_noise_2)


hdb3 = HDBSCAN(min_cluster_size=5,                            # default
                            min_samples=10,                # default
                            cluster_selection_epsilon=2.24,
                            cluster_selection_method='leaf',
                            allow_single_cluster=False,      # default
                            metric='euclidean',              # default
                            algorithm='best',
                            leaf_size=20)
y_HDBScan3 = hdb3.fit_predict(X3)       # labels = hdb.labels_

df_hdb3 = X3.copy()
df_hdb3[2] = pd.Series(y_HDBScan3)
centroids3 = df_db3.groupby(2).mean()

n_clusters_3 = len(set(y_HDBScan3)) - (1 if -1 in y_HDBScan3 else 0)
n_noise_3 = list(y_HDBScan3).count(-1)

print("Estimated number of clusters: %d" % n_clusters_3)
print("Estimated number of noise points: %d" % n_noise_3)

#%%

plt.scatter(X[0], X[1], c=y_HDBScan)
plt.colorbar(label='Cluster') 
plt.scatter(centroids.iloc[1:, [0]], centroids.iloc[1:, [1]], marker='D', s=30, c='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('HDBSCAN Cluster')
plt.show()

plt.scatter(X2[0], X2[1], c=y_HDBScan2)
plt.colorbar(label='Cluster') 
plt.scatter(centroids2.iloc[:, [0]], centroids2.iloc[:, [1]], marker='D', s=30, c='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('HDBSCAN Cluster')
plt.show()

plt.scatter(X3[0], X3[1], c=y_HDBScan3)
plt.colorbar(label='Cluster') 
plt.scatter(centroids3.iloc[:, [0]], centroids3.iloc[:, [1]], marker='D', s=30, c='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('HDBSCAN Cluster')
plt.show()

#%%
print(f'Silhouette Score(HDBSCAN n=7): {silhouette_score(X, y_HDBScan)}')
print(f'Silhouette Score(HDBSCAN n=3): {silhouette_score(X2, y_HDBScan2)}')
print(f'Silhouette Score(HDBSCAN n=3): {silhouette_score(X3, y_HDBScan3)}')

#%%
silhuoette_plot(X, y_HDBScan, 'HDBSCAN silhouette plot')
silhuoette_plot(X2, y_HDBScan2, 'HDBSCAN silhouette plot')
silhuoette_plot(X3, y_HDBScan3, 'HDBSCAN silhouette plot')

#%%
cros_table(y_HDBScan, y)
cros_table(y_HDBScan2, y2)
cros_table(y_HDBScan3, y3)

#%%
df['hdbScan'] = change_lab(y_HDBScan, {0: 2, 1: 1, 2: 5, 3: 7, 4: 4, 5: 3, 6:6, -1:0})
df2['hdbScan'] = change_lab(y_HDBScan2, {0: 1, 1: 3, 2: 2, -1:0})
df3['hdbScan'] = change_lab(y_HDBScan3, {0: 2, 1: 1, 2:3, -1: 0})

#%%
best_model = {'gmm': None, 'score': 0, 'y_GMM': []}
best_model2 = {'gmm': None, 'score': 0, 'y_GMM': []}
best_model3 = {'gmm': None, 'score': 0, 'y_GMM': []}

for i in range(1,101):
    gmm = GaussianMixture(n_components=7, n_init=10)
    y_GMM = gmm.fit_predict(X)
    score = silhouette_score(X, y_GMM)
    print(f'Silhouette Score(GaussianMixture n=7): {score}')
    if best_model['score'] < score:
        best_model['score'] = score
        best_model['gmm'] = gmm
        best_model['y_GMM'] = y_GMM
        
for i in range(1,101):
    gmm = GaussianMixture(n_components=3, n_init=10)
    y_GMM2 = gmm.fit_predict(X2)
    y_GMM3 = gmm.fit_predict(X3)
    score2 = silhouette_score(X2, y_GMM2)
    score3 = silhouette_score(X3, y_GMM3)
    print(f'Silhouette Score(GaussianMixture n=3): {score}')
    if best_model2['score'] < score2:
        best_model2['score'] = score2
        best_model2['gmm'] = gmm
        best_model2['y_GMM'] = y_GMM2
    if best_model3['score'] < score3:
        best_model3['score'] = score3
        best_model3['gmm'] = gmm
        best_model3['y_GMM'] = y_GMM3

#%%  Gaussian Mixture

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


df_gmm2 = X2.copy()
df_gmm2[2] = pd.Series(best_model2['y_GMM'])
centroids2 = df_gmm2.groupby(2).mean()

plt.scatter(X2[0], X2[1], c=best_model2['y_GMM'])
plt.colorbar(label='Cluster') 
plt.scatter(centroids2[0], centroids2[1], marker='D', s=30, c='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Gaussian Mixture Cluster')
plt.show()


df_gmm3 = X3.copy()
df_gmm3[2] = pd.Series(best_model3['y_GMM'])
centroids3 = df_gmm3.groupby(2).mean()

plt.scatter(X3[0], X3[1], c=best_model3['y_GMM'])
plt.colorbar(label='Cluster') 
plt.scatter(centroids3[0], centroids3[1], marker='D', s=30, c='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Gaussian Mixture Cluster')
plt.show()

#%%
print(f'Silhouette Score(Gaussian Mixture n=7): {best_model["score"]}')
print(f'Silhouette Score(Gaussian Mixture n=3): {best_model2["score"]}')
print(f'Silhouette Score(Gaussian Mixture n=3): {best_model3["score"]}')

#%%
silhuoette_plot(X, best_model['y_GMM'], 'Gaussian Mixture silhouette plot')
silhuoette_plot(X2, best_model2['y_GMM'], 'Gaussian Mixture silhouette plot')
silhuoette_plot(X3, best_model3['y_GMM'], 'Gaussian Mixture silhouette plot')

#%%
cros_table(best_model['y_GMM'], y)
cros_table(best_model2['y_GMM'], y2)
cros_table(best_model3['y_GMM'], y3)

#%%
df['GaussianMix'] = change_lab(best_model['y_GMM'], {0: 7, 1: 6, 2: 2, 3: 3, 4: 4, 5: 5, 6:1})
df2['GaussianMix'] = change_lab(best_model2['y_GMM'], {0: 3, 1: 1, 2: 2})
df3['GaussianMix'] = change_lab(best_model3['y_GMM'], {0: 3, 1: 2, 2: 1})

#%%
def getScores(data, y_kmean, y_db, y_hdb, model_bodel_num):
    kmeans_precision = precision_score(data[2], data['kMeans'],average = "macro")
    kmeans_recall = recall_score(data[2], data['kMeans'],average = "macro")
    kmeans_silhouette = silhouette_score(data[[0,1]], y_kmean)

    db_precision = precision_score(data[2], data['dbScan'],average = "macro")
    db_recall =recall_score(data[2], data['dbScan'],average = "macro")
    db_silhouette = silhouette_score(data[[0,1]], y_db)

    hdb_precision = precision_score(data[2], data['hdbScan'],average = "macro")
    hdb_recall = recall_score(data[2], data['hdbScan'],average = "macro")
    hdb_silhouette = silhouette_score(data[[0,1]], y_hdb)

    gmm_precision = precision_score(data[2], data['GaussianMix'],average = "macro")
    gmm_recall = recall_score(data[2], data['GaussianMix'],average = "macro")
    gmm_silhouette = silhouette_score(data[[0,1]], model_bodel_num['y_GMM'])

    df_scores = {
        "KMEANS": [kmeans_precision, kmeans_recall, kmeans_silhouette],
        "DBSCAN": [db_precision, db_recall, db_silhouette],
        "HDBSCAN": [hdb_precision, hdb_recall, hdb_silhouette],
        "GMM": [gmm_precision, gmm_recall, gmm_silhouette]
    }
    index = ["Precision", "Recall", "Silhouette"]
    result_df = pd.DataFrame(df_scores, index=index)

    # Display the DataFrame
    print(result_df)
    return result_df

#%%
getScores(df, y_mean, y_DBScan, y_HDBScan, best_model)
getScores(df2, y_mean2, y_DBScan2, y_HDBScan2, best_model2)
getScores(df3, y_mean3, y_DBScan3, y_HDBScan3, best_model3)
