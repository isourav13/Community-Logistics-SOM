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
som = MiniSomCustom(som_shape[0], som_shape[1], train_data.shape[1], sigma=.5, learning_rate=.5,
              neighborhood_function='gaussian', activation_distance='euclidean', random_seed=10)


############################   TRAIN   ##################################

som.train_batch(train_data, 500, verbose=True)

############################   TEST   ##################################

# each neuron represents a cluster
winner_coordinates = np.array([som.winner(x) for x in test_data]).T
# with np.ravel_multi_index we convert the bidimensional coordinates to a monodimensional index
cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)

############################   Figure 1 ##################################


font1 = {'family':'serif','color':'black','size':20}
font2 = {'family':'serif','color':'darkred','size':15}

plt.figure(figsize=(6, 6))
for c in np.unique(cluster_index):
    plt.scatter(test_data[cluster_index == c, 3],
                test_data[cluster_index == c, 4], label='Cluster='+str(c+1), alpha=.7)

# plotting centroids
for centroid in som.get_weights():
    plt.scatter(centroid[:, 0], centroid[:, 1], marker='x',
                s=5, linewidths=10, color='k', label='centroid')

plt.legend();
plt.title("Self Organising Map - Clustering", fontdict = font1)
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
x= test_data[0:20,3]
y= test_data[0:20,4]
som = MiniSomCustom(som_shape[0], N_neurons, 2, sigma=2, learning_rate=.4, neighborhood_function='gaussian', random_seed=0)
points = np.array([x,y]).T
som.random_weights_init(points)


plt.figure(figsize=(10,10))
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
plt.scatter(test['Latitude'], test['Longitude'], c= kmeans.labels_.astype(float), s=50, alpha=0.5, linewidths=0.2)
plt.scatter(centroids[:, 3], centroids[:, 4], c='red', s=200, marker='x')
plt.show()




############################   KPI and Numerical Analysis    ##################################


K_means_dvs=metrics.davies_bouldin_score(test, cluster_index)
K_means_si=metrics.silhouette_score(test, cluster_index, metric='euclidean')
K_means_chi=metrics.calinski_harabasz_score(test, cluster_index)

SOM_dvs=metrics.davies_bouldin_score(test, labels)
SOM_si=metrics.silhouette_score(test, labels, metric='euclidean')
SOM_chi=metrics.calinski_harabasz_score(test, labels)

print(f'Davies-Bouldin Index: SOM_dvs={SOM_dvs} & K_means_dvs={K_means_dvs}')
print(f'Silhouette Coefficient: SOM_si={SOM_si} & K_means_si={K_means_si}')
print(f'Calinski-Harabasz Index: SOM_chi={SOM_chi} & K_means_chi={K_means_chi}')

# Davies-Bouldin Index: SOM_dvs=0.7869547436438555 & K_means_dvs=4.095046246890271
# Silhouette Coefficient: SOM_si=0.42874949024853665 & K_means_si=0.04004837828245745
# Calinski-Harabasz Index: SOM_chi=828.0990947535752 & K_means_chi=271.7028326944662

plt.title("Davies-Bouldin Index")
x = ['SOM', 'K-Means']
y   = [SOM_dvs, K_means_dvs]
plt.bar(x, y, width=0.50, edgecolor='k', linewidth=2, align='center', color='green')
plt.xlabel("Clustering Type")
plt.ylabel("Score")
plt.yticks(ticks=[x * 1 for x in range(10)])
plt.show()

plt.title("Silhouette Coefficient")
x = ['SOM', 'K-Means']
y   = [SOM_si, K_means_si]
plt.bar(x, y, width=0.50, edgecolor='k', linewidth=2, align='center')
plt.xlabel("Clustering Type")
plt.ylabel("Score")
plt.yticks(ticks=[x * 0.1 for x in range(10)])
plt.show()

plt.title("Calinski-Harabasz Index")
x = ['SOM', 'K-Means']
y   = [SOM_chi, K_means_chi]
plt.bar(x, y, width=0.50, edgecolor='k', linewidth=2, align='center', color='blue')
plt.xlabel("Clustering Type")
plt.ylabel("Score")
plt.yticks(ticks=[x * 100 for x in range(10)])
plt.show()
