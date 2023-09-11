import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import load_npz

n_clusters = 15
print("Pre-clustering users with K-means with", n_clusters, "clusters")
      
print("Getting the csr matrix...")
csr_util_mat = load_npz("./data/processed/csr_ratings.npz")
print("Getting the utility matrices...")
pivot =  pd.read_csv('./data/processed/pivot.csv')
print("Getting the ratings matrix...")
ratings = pd.read_csv('./data/processed/final_ratings.csv')
      
kmeans = KMeans(init='random', n_clusters=5, algorithm='lloyd', n_init='auto').fit(csr_util_mat)
kmeans_labels = kmeans.labels_ 
unique_labels, label_counts = np.unique(kmeans_labels, return_counts=True)

print("The number of users per class:\n")
for u, c in zip(unique_labels, label_counts):
    print(u, c)
    
#TODO: fix very small clusters

print("Creating rating matrices for each cluster...")
ratings_clusters = dict()
for l in range(len(unique_labels)):
    # find userId of users in each cluster
    uindxs = np.where(kmeans_labels == l)[0]

    filter_arr = []
    for i in np.array(pivot.index):
        if i in uindxs:
              filter_arr.append(True)
        else:
              filter_arr.append(False)
    uids = pivot[filter_arr].userId.to_list()
    #create sub-matrices of ratings
    mask = ratings['userId'].isin(uids)
    df = ratings[mask]
    ratings_clusters[l] = df.reset_index(drop=True)
    
#find which clasture and which matrix corresponds to the user
def getUserClusterByUserId(userId):
    l = kmeans_labels[pivot[pivot.userId == userId].index]
    return l[0]