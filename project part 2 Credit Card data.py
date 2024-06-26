# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:35:11 2024

@author: Mathi
"""


import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

random.seed(123)
#%%


df = pd.read_csv("CC_GENERAL.csv")
df.fillna(0, inplace=True)
scaler = StandardScaler()

df_scaled = pd.DataFrame(scaler.fit_transform(df.iloc[:,1:18]), index=df.iloc[:,1:18].index, columns=df.iloc[:,1:18].columns)


a = df_scaled["BALANCE"]
b = df_scaled["BALANCE_FREQUENCY"]
c = df_scaled["PURCHASES"]
d = df_scaled["ONEOFF_PURCHASES"]
e = df_scaled["INSTALLMENTS_PURCHASES"]
f = df_scaled["CASH_ADVANCE"]
g = df_scaled["PURCHASES_FREQUENCY"]
h = df_scaled["ONEOFF_PURCHASES_FREQUENCY"]
i = df_scaled["PURCHASES_INSTALLMENTS_FREQUENCY"]
j = df_scaled["CASH_ADVANCE_FREQUENCY"]
k = df_scaled["CASH_ADVANCE_TRX"]
l = df_scaled["PURCHASES_TRX"]
m = df_scaled["CREDIT_LIMIT"]
n = df_scaled["PAYMENTS"]
o = df_scaled["MINIMUM_PAYMENTS"]
p = df_scaled["PRC_FULL_PAYMENT"]
q = df_scaled["TENURE"]
data = list(zip(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q))

#%%



#%%
#Kmeans model

best_model_kmeans = None
best_score_kmeans = -1  # Initialize with a value lower than any possible silhouette score
for i in range(2,5):
    for _ in range(10):
        # Clustering with Kmeans
        cluster = i
        kmeans = KMeans(n_clusters=cluster,n_init=10)
        kmeans.fit(data)
        
    
        
        # Compute silhouette score
        score = silhouette_score(data, kmeans.fit_predict(data))
        print(score)
        # Check if the current model has a higher silhouette score than the best model so far
        if score > best_score_kmeans:
            best_model_kmeans = kmeans
            best_score_kmeans = score
            clusters_kmeans = i
            labels = kmeans.labels_
            df['cluster_Kmeans']=labels
    
                


print("The highest silhouette score found for KMeans model:", best_score_kmeans,"Happens with ",clusters_kmeans, "clusters")



#%%
#Trying to find the best GMM plot
best_model_GMM = None
best_score_GMM = -1  # Initialize with a value lower than any possible silhouette score
for i in range(2,5):
    for _ in range(10):
        # Clustering with Gaussian Mixture Model
        num_clusters = i 
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
            clusters = i
            df['clusters_GMM']=labels
    


# After n iterations, print the silhouette score of the best model
print("The highest silhouette score found for Gaussian Mixture Model:", best_score_GMM,"Happens with ",clusters, "clusters")
#%%


#We choose to use the kmeans clustering, as it has the highest silhouete score

#Since it is not possible to visualize a 18D scatterplot we have to reduce the amount of dimensions we are looking at

#To do that, we are going to use Principal Component Analysis (PCA) to do so.



#PCA


plt.figure(figsize=(20, 10))
sns.heatmap(df_scaled.corr(), annot=True, cmap='viridis')



pca = PCA(n_components=5)
pca.fit(df_scaled)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_[:2].sum())


# Visual for each component’s explained variance
fig, ax = plt.subplots(figsize=(6,6))
ax.plot(pca.explained_variance_ratio_,"bo--",linewidth=2)
ax.set_xlabel("Principal Components", fontsize = 12)
ax.set_ylabel("Explained Variance", fontsize = 12)
ax.set_title("Explained Variance Ratio", fontsize = 16)
n = len(pca.explained_variance_ratio_)
plt.xticks(np.arange(n), np.arange(1, n+1));


# We see, that over 50% of the variance can be explained by just using two components

# Making a dataframe with principal components
pca2 = PCA(n_components=2)
pca2.fit(df_scaled)
principalComponents = pca2.fit_transform(df_scaled)

df_pca = pd.DataFrame(data = principalComponents, columns = ["principal component 1", "principal component 2"])
df_pca = pd.concat([df_pca, df[["cluster_Kmeans"]]], axis = 1)
df_pca

colors = ['blue' if value == 1 else 'red' if value == 2 else 'yellow' for value in df_pca['cluster_Kmeans']]
plt.scatter(df_pca.iloc[:, 0], df_pca.iloc[:, 1],c=colors)



plt.xlabel('principal component 1')
plt.ylabel('principal component 2')
plt.title('KMEANS clustering using PC1 and PC2 as x and y axis')
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='Cluster 0', markerfacecolor='yellow', markersize=8),
    plt.Line2D([0], [0], marker='o', color='w', label='Cluster 1', markerfacecolor='blue', markersize=8),
    plt.Line2D([0], [0], marker='o', color='w', label='Cluster 2', markerfacecolor='red', markersize=8)
]
plt.legend(handles=legend_elements)

# Show the plot
plt.show()



#%%
#Kmeans model

#We could also have done the kmeans model on just the PCA to reduce computation time.
#This however comes with the loss of some of the explaniable variance.

best_model_kmeans_pca = None
best_score_kmeans_pca = -1  # Initialize with a value lower than any possible silhouette score
x_ = df_pca["principal component 1"]
y_ = df_pca["principal component 2"]
data_pca = list(zip(x_, y_))

for _ in range(50):
    # Clustering with Kmeans
    cluster = 3
    kmeans = KMeans(n_clusters=cluster,n_init=10)
    kmeans.fit(data_pca)
    

    
    # Compute silhouette score
    score = silhouette_score(data_pca, kmeans.fit_predict(data_pca))
    print(score)
    # Check if the current model has a higher silhouette score than the best model so far
    if score > best_score_kmeans:
        best_model_kmeans_pca = kmeans
        best_score_kmeans_pca = score


plt.scatter(x_, y_, c=best_model_kmeans_pca.predict(data_pca))
plt.title("Clusters made with the best kmeans model using only PC 1 and 2")
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
print("The highest silhouette score found for KMeans model:", best_score_kmeans)



#%%
sample_silhouette_values_kmeans = silhouette_samples(data, best_model_kmeans.predict(data))

import matplotlib.cm as cm
y_lower = 10
n_clusters = 3
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
for i in range(3):
    # Aggregate the silhouette scores for samples belonging to
    # cluster i, and sort them
    ith_cluster_silhouette_values = sample_silhouette_values_kmeans[df['cluster_Kmeans'] == i]

    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.nipy_spectral(float(i) / n_clusters)
    ax1.fill_betweenx(
        np.arange(y_lower, y_upper),
        0,
        ith_cluster_silhouette_values,
        facecolor=color,
        edgecolor=color,
        alpha=0.7,
    )

    # Label the silhouette plots with their cluster numbers at the middle
    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))



ax1.set_title("The silhouette plot for the various clusters in KMEANS.")
ax1.set_xlabel("The silhouette coefficient values")
ax1.set_ylabel("Cluster label")

# The vertical line for average silhouette score of all the values
ax1.axvline(x=best_score_kmeans, color="red", linestyle="--")

ax1.set_yticks([])  # Clear the yaxis labels / ticks
ax1.set_xticks([-0.2,-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
#%%

sample_silhouette_values_gmm = silhouette_samples(data, best_model_GMM.predict(data))

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
for i in range(2):
    # Aggregate the silhouette scores for samples belonging to
    # cluster i, and sort them
    ith_cluster_silhouette_values = sample_silhouette_values_gmm[df['clusters_GMM'] == i]

    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.nipy_spectral(float(i) / 2)
    ax1.fill_betweenx(
        np.arange(y_lower, y_upper),
        0,
        ith_cluster_silhouette_values,
        facecolor=color,
        edgecolor=color,
        alpha=0.7,
    )

    # Label the silhouette plots with their cluster numbers at the middle
    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))


ax1.set_title("The silhouette plot for the various clusters in GMM.")
ax1.set_xlabel("The silhouette coefficient values")
ax1.set_ylabel("Cluster label")

# The vertical line for average silhouette score of all the values
ax1.axvline(x=best_score_GMM, color="red", linestyle="--")

ax1.set_yticks([])  # Clear the yaxis labels / ticks
ax1.set_xticks([-0.5,-0.25, 0, 0.2, 0.4, 0.6, 0.8, 1])

#%%

best_model_kmeans_pca.save('best_model_kmeans_pca.keras')
best_model_GMM.save('best_model_GMM.keras')
best_model_kmeans.save('ModelNN.keras')

