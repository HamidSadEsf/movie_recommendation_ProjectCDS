import pandas as pd
import numpy as np
import time
import random
from scipy.sparse import csr_matrix, save_npz

# import datasets:
print("Creating dataframes...")
genome_scores =  pd.read_csv('data/external/genome-scores.csv')
movies = pd.read_csv('data/external/movies.csv',  usecols = ["movieId"])
ratings = pd.read_csv('data/external/ratings.csv', usecols = ["movieId", "userId", "rating"])
sveta_ratings = pd.read_csv('data/external/sveta-ratings.csv')
hamid_ratings = pd.read_csv('data/external/hamid-ratings.csv')

#create anti sveta and anti-hamid ratings
anti_sveta_ratings = sveta_ratings.copy()
anti_sveta_ratings['rating'] = sveta_ratings["rating"].apply(lambda x: -1 * (x - sveta_ratings.rating.mean())+ sveta_ratings.rating.mean())
anti_hamid_ratings = hamid_ratings.copy()
anti_hamid_ratings['rating'] = hamid_ratings["rating"].apply(lambda x: -1 * (x - hamid_ratings.rating.mean())+ hamid_ratings.rating.mean())

# Create a movie - user - rating data frame with movies present in all the dataframes
print("Filtering users...")
tagged_movies = pd.DataFrame(genome_scores['movieId'].value_counts()).index
mov_rat = pd.merge(movies, ratings, on="movieId")
final_df = mov_rat[mov_rat["movieId"].isin(tagged_movies)]

#Shringking users
udf = pd.DataFrame(final_df['userId'].value_counts())
lazy_users = udf[udf['userId'] <= 60].index
final_df = final_df[~final_df["userId"].isin(lazy_users)]
print('Out of total of ', udf.shape[0] , ' users, ', lazy_users.shape[0], ' are considered lazy and will be removed.')
print('The final number of users is ', final_df["userId"].nunique())

#Randomly choose N users 
selection_of_users = random.sample(final_df["userId"].value_counts().index.to_list(), 400)
final_df = final_df[final_df["userId"].isin(selection_of_users)]
print('Randomly choosing', len(selection_of_users), 'users...')

def add_user_to_ratings(new_user_ratings, name, ratings):
    new_user_id = random.choice(list(set([x for x in range(ratings.userId.min(),ratings.userId.max())]) - set(ratings.userId.values)))
    df = new_user_ratings[["movie_id", "rating"]].rename(columns={"movie_id": "movieId"})
    df["userId"] = new_user_id
    ratings = pd.concat([df, ratings])
    print(name, "'s user id is", new_user_id)
    return ratings, new_user_id

#Add Sviatlana's and Hamid's ratings
print("Adding Sviatlana's and Hamid's ratings... we are lazy users, but still...")
final_df, sveta_user_id = add_user_to_ratings(sveta_ratings, "Sveta", final_df)
final_df, hamid_user_id = add_user_to_ratings(hamid_ratings,"Hamid", final_df)

#Add anti-Sviatlana's and anti-Hamid's ratings
print("Adding anti-Sviatlana's and anti-Hamid's ratings... they are lazy users, but still...")
final_df, anti_sveta_user_id = add_user_to_ratings(anti_sveta_ratings, "anti-Sveta", final_df)
final_df, anti_hamid_user_id = add_user_to_ratings(anti_hamid_ratings,"anti-Hamid", final_df)

print('Now, the final number of users is ', final_df["userId"].nunique())

#Reducing the dataframe by removing unpopular movies and inactive users
#Shringking movies
print("Filtering movies...")
mdf = pd.DataFrame(final_df['movieId'].value_counts())
rare_movies = mdf[mdf['movieId'] <= 10].index
final_df = final_df[~final_df["movieId"].isin(rare_movies)]
print('Out of total of ', mdf.shape[0] , ' movies, ', rare_movies.shape[0], ' are considered rare and will be removed.')
print('The final number of movies is ', final_df.groupby(by="movieId").count().shape[0])

# Create the user->movie sparse rating matrix.
print("Creating the pivot matrix...")
pivot = final_df.pivot_table(index="userId", columns="movieId", values="rating")
pivot_na = pivot.copy()

# Lets save the pivot matrix with NA for further uses
#pivot_na.to_csv('./data/processed/pivot_na.csv', index=True, header="userId")

#Estimate sparsity
sparsity = 1.0 - ( np.count_nonzero(~np.isnan(pivot)) / float(pivot.size) )
print("The resulting sparcity of the matrix is:", sparsity)

# Fill the NA with zeros
#pivot.fillna(0,inplace=True)
#pivot.to_csv('matrices/pivot.csv', index=True, header="userId")

#Create non-sparse dataset
#csr_data = csr_matrix(pivot.values)
#save_npz("matrices/sparse_ratings.npz", csr_data)

# preparation of the final rating dataframe
print("Preparing the final rating matrix...")
final_ratings = final_df[["movieId", "userId", "rating"]].reset_index(drop=True)

final_ratings.to_csv('./data/processed/final_ratings.csv', index=False)
      
print("Data preprocesssing for collaborative filtering modeling is completed!")