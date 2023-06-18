import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
df_gscore = pd.read_csv('genome-scores.csv')
df_gtags = pd.read_csv('genome-tags.csv')
df_links = pd.read_csv('links.csv')
df_movies = pd.read_csv('movies.csv')
df_rating = pd.read_csv('ratings.csv')
df_tagspd = pd.read_csv('tags.csv')

## Creating the genome dataset

df_gtagscore=df_gtags.merge(df_gscore, how='right', on='tagId').drop('tag', axis =1)
df_gtagscore.drop_duplicates(subset=['tagId', 'movieId'], inplace=True)

# pivoting the datafram
df_gtagscore = df_gtagscore.pivot(index = 'movieId', columns = 'tagId', values='relevance')

## adding realease year and genres as columns

df_movies.genres= df_movies.genres.str.split('|')
dummies = pd.get_dummies(df_movies.genres.apply(pd.Series).stack()).sum(level=0)
df_movies = pd.concat([df_movies, dummies], axis=1).drop('genres', axis=1)

from sklearn.preprocessing import MinMaxScaler
df_movies.drop('title', axis =1)

df_movies = pd.read_csv('movies.csv')
mask = df_movies['title'].str.contains('09–')

df_movies = pd.read_csv('movies.csv')
def condition(x):
    if x[-2:]=='a)':
        return np.nan
    elif x[-2:]=='l)':
        return np.nan
    elif x[-3:-1]=='7-':
        return 2007
    elif x[-4:-2]=='9–':
        return 2009
    elif x[-2:]=='))':
        return x[-6:-2]
    elif x[-1:]==')':
        return x[-5:-1]
    elif x[-1:]==' ':
        return x[-6:-2]
    else:
        return np.nan
df_movies['releaseyear'] = df_movies['title'].apply(condition).fillna(1993)

scaler02 = MinMaxScaler()
df_movies.releaseyear = scaler02.fit_transform(df_movies[['releaseyear']])

df_ContBaseRec = pd.merge(df_gtagscore, df_movies, how='inner', on='movieId').set_index('movieId')

df_ContBaseRec.info()

#Clsutering
#importing the library

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

## training the Model

kmeans = KMeans(n_clusters=18, n_init='auto')
kmeans.fit(df_ContBaseRec)
labels = kmeans.labels_ # getting the movis labled with the corresponding cluster.
#making a new df of the movies with the labels and MovieId, title and release year.
df_labeledMovies= pd.DataFrame({'movieId': df_ContBaseRec.index, 'labels': labels})\
                    .merge(df_movies, on='movieId').set_index('movieId')

# recommending the nearest neighbors within the given cluster with the example "Network (1975)

from sklearn.neighbors import NearestNeighbors

#specifying the movie
given_movie_id =df_labeledMovies[df_labeledMovies.title.str.contains(r'\bNetwork\b.{0,9}$', case=False)].labels.item()
given_movie_cluster = df_labeledMovies.loc[given_movie_id, "labels"] # getting the cluster
movies_in_cluster_indices = df_labeledMovies[df_labeledMovies["labels"] == given_movie_cluster].index.tolist()
movies_in_cluster_relevance = df_ContBaseRec.loc[movies_in_cluster_indices] # Filter the relevance data based on movies in the same cluster

# Train the Nearest Neighbors algorithm
nn = NearestNeighbors(n_neighbors=6)  # 5 neighbors + the given movie itself
nn.fit(movies_in_cluster_relevance)
distances, indices = nn.kneighbors([df_ContBaseRec.loc[given_movie_id]]) # finding the nearest neigbors
nearest_neighbor_ids = [movies_in_cluster_indices[i] for i in indices[0]] # Get the IDs of the nearest neighbors
nearest_neighbor_ids.remove(given_movie_id) # Remove the given movie from the list of neighbors

df_labeledMovies.loc[nearest_neighbor_ids]