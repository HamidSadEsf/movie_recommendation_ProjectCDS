import pandas as pd
from sklearn.cluster import KMeans

# training the Model
def get_labeledMovies(df, clusters=5):
    kmeans = KMeans(n_clusters=clusters, n_init='auto')
    kmeans.fit(df)
    labels = kmeans.labels_
    df_movies = pd.read_csv('data/movies.csv')
    df_labeledMovies= pd.DataFrame({'movieId': df.index, 'labels': labels}).merge(df_movies, on='movieId').set_index('movieId')
    return df_labeledMovies