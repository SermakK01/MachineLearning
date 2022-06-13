from sklearn.datasets import fetch_openml
import numpy as np
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
mnist.target = mnist.target.astype(np.uint8)
X = mnist["data"]
y = mnist["target"]

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

score = []
for k in range (8,13):
    kmeans = KMeans(n_clusters=k)
    y_pred = kmeans.fit_predict(X)
    score.append(silhouette_score(X, y_pred))

kmeans_10 = KMeans(n_clusters=10)
y_pred_10 = kmeans.fit_predict(X)

import pickle

with open('kmeans_sil.pkl', 'wb') as f:
    pickle.dump(score,f)

import pandas as pd
from sklearn.metrics import confusion_matrix
from pandas import Series

matrix = confusion_matrix(y, y_pred_10)

max_arg = []

for row in matrix:
    max_arg.append(np.argmax(row))

max_arg.sort()

with open('kmeans_argmax.pkl', 'wb') as f:
    pickle.dump(max_arg,f)

result = []
for x in range(0, 300):
    for i in range(0, len(X) - 1):
        result.append(np.linalg.norm(X[x] - X[i]))

result.sort()
result = result[300:]

result = result[:10]

with open('dist.pkl', 'wb') as f:
    pickle.dump(result,f)

s = (result[1]+result[2]+result[0])/3

i = s
list1 = []
while (i < s + s * 0.10):
    list1.append(i)
    i += i * 0.04


list2 = []
from sklearn.cluster import DBSCAN
for i in list1:
    dbscan = DBSCAN(eps=i)
    dbscan.fit(X)
    list2.append(np.ma.count(np.unique(dbscan.labels_)))

with open('dbscan_len.pkl', 'wb') as f:
    pickle.dump(list2,f)

