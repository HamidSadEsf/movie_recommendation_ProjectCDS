# Import modules

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, load_npz
from scipy import sparse
from sklearn import neighbors

# load sparce utility matrix
csr_util_mat = load_npz("matrices/sparse_ratings.npz")

def keep_rows_csr(mat, indices):
    """
   Keep the rows denoted by ``indices`` form the CSR sparse matrix ``mat``.
    """
    #if not isinstance(mat, scipy.sparse.csr_matrix):
     #   raise ValueError("works only for CSR format -- use .tocsr() first")
    indices = indices.flatten()
    mask = np.zeros(mat.shape[0], dtype=bool)
    mask[indices] = True
    return mat[mask]
    

def closest_users_mean_ratings(mat, movies):
    """
    Aggregating the ratings from the closest users by calculating the averaged rating of the movies given by the neiboghrs
    """
    mat_array = mat.toarray()
    mat_array[mat_array == 0] = np.nan
    av_ratings = np.nanmean(mat_array, axis=0)
    df = pd.DataFrame(data ={'movieId': movies, 'rating': av_ratings })
    
    return df


def get_predictions(user_id, util_mat, n_predictions):
    """
    Return predictions of the collaborative filtering system, where ``n_predictions`` is the desired number of predictions
    """
    # Calculate the userId index in the sparse matrix
    user_index = np.where(util_mat.index == user_id)[0][0]
    
    # Nearest Neighbors
    number_of_closest_users = 150
    nn = neighbors.NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=number_of_closest_users)
    nn.fit(csr_util_mat) 

    # Find the nearest neighbors for the target user (e.g., User1)

    mask = np.zeros(csr_util_mat.shape[0], dtype=bool)
    mask[user_index] = True
    target_user_row = csr_util_mat[mask]

    distances, indices = nn.kneighbors(target_user_row)
    
    csr_data_closests_users = keep_rows_csr(csr_util_mat, indices)
    rating_aggregation = closest_users_mean_ratings(csr_data_closests_users, util_mat.columns)

    return rating_aggregation
