# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 22:29:08 2019

@author: VINIT KORADE
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections

data = pd.read_csv("C:\\Users\\ADMIN\\Desktop\\LP3\\ML\\Exp4\\data.csv")
x = data.iloc[:,0].values
y = data.iloc[:,1].values

init_centroid = np.array([[0.1, 0.6], [0.3, 0.2]])
print("\nInitial Centroids:- ",init_centroid)

from sklearn.cluster import KMeans
clusters = KMeans(n_clusters=2, init= init_centroid)
clusters.fit(data)

centroid = clusters.cluster_centers_
print("\nFinal Centroids:- ",centroid)

print("\nPoint P6 belongs to cluster: ",clusters.labels_[5])

population= collections.Counter(clusters.labels_)
print("\nPopulation of cluster m2: ",population[1])
print("\n")

plt.scatter(x, y, c= clusters.labels_.astype(float), s=50, alpha=1)
plt.scatter(clusters.cluster_centers_[:,0], clusters.cluster_centers_[:,1], c='red', s=50)
