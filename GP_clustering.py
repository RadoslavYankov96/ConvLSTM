import pandas as pd
from sklearn.cluster import KMeans
import os


df = pd.read_csv('/home/itsnas/ueuua/BA/dataset/dataset_info.csv')
kmeans = KMeans(n_clusters=3, random_state=42).fit(df[['HS 1', 'HS 2', 'HS 3']])
centroids = kmeans.cluster_centers_
labels = kmeans.labels_
print(centroids)
print(labels)
df["cluster"] = labels
os.chdir('/home/itsnas/ueuua/BA/dataset')
df.to_csv('kmeans_cluster.csv')
