import pandas as pd
from sklearn.cluster import KMeans

# training the Model
def get_labeledMovies():
    from Preprocess_Content_Based import get_df
    kmeans = KMeans(n_clusters=18, n_init='auto')
    df_ContBaseRec = get_df()
    kmeans.fit(df_ContBaseRec)
    labels = kmeans.labels_
    df_movies = pd.read_csv('movies.csv')
    df_labeledMovies= pd.DataFrame({'movieId': df_ContBaseRec.index, 'labels': labels}).merge(df_movies, on='movieId').set_index('movieId')
    return df_labeledMovies, df_ContBaseRec