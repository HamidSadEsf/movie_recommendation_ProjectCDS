import pandas as pd
import numpy as np
import time
from scipy.sparse import csr_matrix, save_npz

# import datasets:
print("Creating dataframes...")
genome_scores =  pd.read_csv('data/external/genome-scores.csv')
movies = pd.read_csv('data/external/movies.csv',  usecols = ["movieId"])
ratings = pd.read_csv('data/external/ratings.csv', usecols = ["movieId", "userId", "rating"])
sveta_ratings = pd.read_csv('data/external/sveta-ratings.csv')
hamid_ratings = pd.read_csv('data/external/hamid-ratings.csv')

# Create a movie - user - rating data frame with movies present in all the dataframes
print("Filtering movies and users...")
tagged_movies = pd.DataFrame(genome_scores['movieId'].value_counts()).index
mov_rat = pd.merge(movies, ratings, on="movieId")
final_df = mov_rat[mov_rat["movieId"].isin(tagged_movies)]

#Reducing the dataframe by removing unpopular movies and inactive users
#Shringking movies
mdf = pd.DataFrame(final_df['movieId'].value_counts())
rare_movies = mdf[mdf['movieId'] <= 500].index
final_df = final_df[~final_df["movieId"].isin(rare_movies)]
print('Out of total of ', mdf.shape[0] , ' movies, ', rare_movies.shape[0], ' are considered rare and will be removed.')
print('The final number of movies is ', final_df["movieId"].nunique())

#Shringking users
udf = pd.DataFrame(final_df['userId'].value_counts())
lazy_users = udf[udf['userId'] <= 500].index
final_df = final_df[~final_df["userId"].isin(lazy_users)]
print('Out of total of ', udf.shape[0] , ' users, ', lazy_users.shape[0], ' are considered lazy and will be removed.')
print('The final number of users is ', final_df["userId"].nunique())

# Create the user->movie sparse rating matrix.
print("Creating the pivot matrix...")
pivot = final_df.pivot_table(index="userId", columns="movieId", values="rating")
pivot_na = pivot.copy()

#Create non-sparce dataset
csr_data = csr_matrix(pivot.values)

# Lets save the pivot matrix with NA for further uses
pivot_na.to_csv('./data/processed/pivot_na.csv', index=True, header="userId")

#Estimate sparsity
sparsity = 1.0 - ( np.count_nonzero(~np.isnan(pivot)) / float(pivot.size) )
print("The resulting sparcity of the matrix is:", sparsity)

# Fill the NA with zeros
#pivot.fillna(0,inplace=True)
#pivot.to_csv('matrices/pivot.csv', index=True, header="userId")

# preparation of the final rating dataframe
print("Preparing the final rating matrix...")
final_ratings = final_df[["movieId", "userId", "rating"]].reset_index(drop=True)
final_ratings.to_csv('./data/processed/final_ratings.csv', index=True)
      
print("Data preprocesssing for collaborative filtering modeling is completed!")