#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 08:41:14 2021

@author: isourav
"""

from numpy.random import RandomState
import pandas as pd
import numpy as np
#from minisom import MiniSom
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import geopandas as gpd
from shapely.geometry import Point, Polygon
import contextily as cx
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.cluster import AgglomerativeClustering
from fcmeans import FCM


import sys
sys.path.insert(0, '/Users/isourav/Documents SSD/Assignment/Dissertation/Code/MiniSom')
from minisommodified import MiniSomCustom

############################   Data Loading and Randomizing weight column w.r.t. Sales per person  ##################################

df = pd.read_csv('/Users/isourav/Documents SSD/Assignment/Dissertation/Dataset/Input_Data.csv')
df['Weight'] = np.random.randint(1, 5, df.shape[0])*df.iloc[:, 1]


############################   Data Preprocessing - Train 70%, Test 30%  ##################################

rng = RandomState()
train = df.sample(frac=0.7, random_state=rng)
test = df.loc[~df.index.isin(train.index)]

############################   NumPy Object Scaling for SOM  ##################################

train_data = (train - np.mean(train, axis=0)) / np.std(train, axis=0)
train_data = train_data.values

test_data = (test - np.mean(test, axis=0)) / np.std(test, axis=0)
test_data = test_data.values

som_shape = (1, 4)
som = MiniSomCustom(som_shape[0], som_shape[1], train_data.shape[1], sigma=.5, learning_rate=.05,
              neighborhood_function='gaussian', activation_distance='euclidean', random_seed=10)


############################   TRAIN   ##################################

som.train_batch(train_data, 2, verbose=True)

############################   TEST   ##################################

# each neuron represents a cluster
winner_coordinates = np.array([som.winner(x) for x in test_data]).T
# with np.ravel_multi_index we convert the bidimensional coordinates to a monodimensional index
cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)

############################   Figure 1 ##################################


font1 = {'family':'serif','color':'black','size':15}
font2 = {'family':'serif','color':'darkred','size':10}

plt.figure(figsize=(6, 6))
for c in np.unique(cluster_index):
    plt.scatter(test_data[cluster_index == c, 3],
                test_data[cluster_index == c, 4], label='Cluster='+str(c+1), alpha=.7)

# plotting centroids
for centroid in som.get_weights():
    plt.scatter(centroid[:, 0], centroid[:, 1], marker='x',
                s=5, linewidths=10, color='k', label='centroid')

plt.legend();
plt.title("WSOM-DRP Clustering", fontdict = font1)
plt.xlabel("Latitude", fontdict = font2)
plt.ylabel("Longitude", fontdict = font2)

############################   GeoDataFrame Plot  ##################################

nyc = gpd.read_file('/Users/isourav/Documents SSD/Assignment/Dissertation/Dataset/Shape/geo_export_900c7e2d-c78c-44d4-8d41-4ee68d2d5b9f.shp')
df_wm=nyc.to_crs(epsg=4326)
crs = {'init':'EPSG:4326'}
geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
geo_df = gpd.GeoDataFrame(df,
                          crs = crs,
                          geometry = geometry)


ax = df_wm.plot(figsize=(10, 10), alpha=0.3, edgecolor='BLACK', linewidth=3)
ax.set_title('New York City')
cx.add_basemap(ax, crs=df_wm.crs.to_string(), source=cx.providers.Stamen.Watercolor, zoom=12)
cx.add_basemap(ax, crs=df_wm.crs.to_string(), source=cx.providers.Stamen.TonerLabels, zoom=10)
geo_df.plot(ax=ax,color='BLUE',linewidth=0.4)

############################   TSP   ##################################


np.random.RandomState(10)
N_points = 12
N_neurons = N_points*2
t = np.linspace(0, np.pi*2, N_points)
x= test_data[0:20,0]
y= test_data[0:20,1]
som = MiniSomCustom(som_shape[0], N_neurons, 2, sigma=2, learning_rate=.4, neighborhood_function='gaussian', random_seed=0)
points = np.array([x,y]).T
som.random_weights_init(points)


plt.figure(figsize=(10,10))
plt.suptitle("Travelling Salesman Problem using FODWSOM", fontdict = font1)

for i, iterations in enumerate(range(10, 601, 50)):
    som.train(points, iterations, verbose=False, random_order=False)
    plt.subplot(3, 4, i+1)
    plt.scatter(x,y)
    visit_order = np.argsort([som.winner(p)[1] for p in points])
    visit_order = np.concatenate((visit_order, [visit_order[0]]))
    plt.plot(points[visit_order][:,0], points[visit_order][:,1])
    plt.title("iterations: {i};\nerror: {e:.3f}".format(i=iterations,
                                                        e=som.quantization_error(points)))
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.show()


############################   K-means    ##################################

kmeans = KMeans(n_clusters=4, algorithm="elkan").fit(test)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_
print(centroids)


plt.figure(figsize=(10,10))
plt.title("k-means - Clustering", fontdict = font1)
plt.xlabel("Latitude", fontdict = font2)
plt.ylabel("Longitude", fontdict = font2)
plt.scatter(test['Latitude'], test['Longitude'], c= kmeans.labels_.astype(float), s=50, alpha=0.5, linewidths=0.2)
plt.scatter(centroids[:, 3], centroids[:, 4], c='red', s=200, marker='x')
plt.show()





hierarchicalClustering = AgglomerativeClustering(n_clusters=4, linkage='ward').fit(test)
labels_hierarchicalClustering = hierarchicalClustering.labels_


plt.figure(figsize=(10,10))
plt.title("Ward Hierarchical Clustering", fontdict = font1)
plt.xlabel("Latitude", fontdict = font2)
plt.ylabel("Longitude", fontdict = font2)
plt.scatter(test['Latitude'], test['Longitude'], c=hierarchicalClustering.labels_,  label='hierarchicalClustering.labels_', cmap='viridis')
plt.show()



fcm = FCM(n_clusters=4)
fc_data=test.values
fcm.fit(fc_data)
fcm_centers = fcm.centers
fcm_labels = fcm.predict(fc_data)

plt.figure(figsize=(10,10))
plt.title("Fuzzy c means Clustering", fontdict = font1)
plt.xlabel("Latitude", fontdict = font2)
plt.ylabel("Longitude", fontdict = font2)
plt.scatter(test['Latitude'], test['Longitude'],  c=fcm_labels, alpha=.1)
plt.scatter(fcm_centers[:,3], fcm_centers[:,4], marker="x", s=500, c='red')
plt.show()


############################   KPI and Numerical Analysis    ##################################


K_means_dvs=metrics.davies_bouldin_score(test, cluster_index)
K_means_si=metrics.silhouette_score(test, cluster_index, metric='euclidean')
K_means_chi=metrics.calinski_harabasz_score(test, cluster_index)


SOM_dvs=metrics.davies_bouldin_score(test, labels)
SOM_si=metrics.silhouette_score(test, labels, metric='euclidean')
SOM_chi=metrics.calinski_harabasz_score(test, labels)


HC_dvs=metrics.davies_bouldin_score(test, labels_hierarchicalClustering)
HC_si=metrics.silhouette_score(test, labels_hierarchicalClustering, metric='euclidean')
HC_chi=metrics.calinski_harabasz_score(test, labels_hierarchicalClustering)

fc_dvs=metrics.davies_bouldin_score(fc_data, fcm_labels)
fc_si=metrics.silhouette_score(fc_data, fcm_labels, metric='euclidean')
fc_chi=metrics.calinski_harabasz_score(fc_data, fcm_labels)




print(f'Davies-Bouldin Index: SOM_dvs={SOM_dvs} & K_means_dvs={K_means_dvs} & HC_dvs={HC_dvs} & fc_dvs={fc_dvs}')
print(f'Silhouette Coefficient: SOM_si={SOM_si} & K_means_si={K_means_si} & HC_si={fc_si} & HC_dvs={fc_si}')
print(f'Calinski-Harabasz Index: SOM_chi={SOM_chi} & K_means_chi={K_means_chi} & HC_chi={HC_chi} & fc_chi={fc_chi}')

# Davies-Bouldin Index: SOM_dvs=0.8001890774559193 & K_means_dvs=2.728198827693249 & HC_dvs=0.8996763503458449 & fc_dvs=0.8361685221182356
# Silhouette Coefficient: SOM_si=0.42345969689340535 & K_means_si=0.05561387873188117 & HC_si=0.40289486682388864 & HC_dvs=0.40289486682388864
# Calinski-Harabasz Index: SOM_chi=813.0363945882436 & K_means_chi=111.75645722003131 & HC_chi=719.5255324368244 & fc_chi=792.2599654651941


plt.figure(figsize=(10,10))
plt.title("Davies-Bouldin Index")
x = ['FODWSOM', 'K-Means']
y   = [SOM_dvs, K_means_dvs]
plt.bar(x, y, width=0.50, edgecolor='k', linewidth=2, align='center', color='green')
plt.xlabel("Clustering Type")
plt.ylabel("Score")
plt.yticks(ticks=[x * 1 for x in range(5)])
plt.show()

plt.figure(figsize=(10,10))
plt.title("Silhouette Index")
x = ['FODWSOM', 'K-Means']
y   = [SOM_si, K_means_si]
plt.bar(x, y, width=0.50, edgecolor='k', linewidth=2, align='center')
plt.xlabel("Clustering Type")
plt.ylabel("Score")
plt.yticks(ticks=[x * 0.1 for x in range(11)])
plt.show()

plt.figure(figsize=(10,10))
plt.title("Calinski-Harabasz Index")
x = ['FODWSOM', 'K-Means']
y   = [SOM_chi, K_means_chi]
plt.bar(x, y, width=0.50, edgecolor='k', linewidth=2, align='center', color='blue')
plt.xlabel("Clustering Type")
plt.ylabel("Score")
plt.yticks(ticks=[x * 100 for x in range(10)])
plt.show()





plt.figure(figsize=(10,10))
plt.title("FODWSOM KPIs")
x = ['SI', 'CHI', 'DVI']
y   = [SOM_si, SOM_chi/20, SOM_dvs]
plt.bar(x, y, width=0.50, edgecolor='k', linewidth=2, align='center', color='brown')
plt.xlabel("KPI Type")
plt.ylabel("Score")
plt.yticks(ticks=[x * 1 for x in range(6)])
plt.show()



plt.figure(figsize=(10,10))
plt.title("k-means KPIs")
x = ['SI', 'CHI', 'DVI']
y   = [K_means_si, K_means_chi/10,K_means_dvs]
plt.bar(x, y, width=0.50, edgecolor='k', linewidth=2, align='center', color='green')
plt.xlabel("KPI Type")
plt.ylabel("Score")
plt.yticks(ticks=[x * 1 for x in range(6)])
plt.show()
