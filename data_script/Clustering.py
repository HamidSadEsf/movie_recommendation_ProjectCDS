import pandas as pd
from sklearn.cluster import KMeans

def get_labeledMovies(df, clusters=5):
    """
    Pre-cluster the movies based on the preprocessed
    Applying clustering by sklearn.cluster.Kmeans and
    get the labels for each item(movie)
    
    Parameters
    ----------
    df:        pandas.DataFrame
        The preprocessed Dataset for the content-based recommendation system
    clusters:  integers
        number of desired clusters (Default is 5) 
    Returns
    ----------
    pandas.DataFrame
        The labeled movies DataFrame 
        Index: movieId
        rows: movies 
        Columns:
            labels:
                integer
                The calculated label (cluster) of each movie    
            title:
                String
                title(release year)
            genres:
                String 
                genres1|genre2|...|genren
            
    """
    #applying Kmeans clustering and getting the labels
    kmeans = KMeans(n_clusters=clusters, n_init='auto')
    kmeans.fit(df)
    labels = kmeans.labels_
    
    # merging with movies dataset to get titles and genres
    df_movies = pd.read_csv('data/movies.csv')
    df_labeledMovies= pd.DataFrame({'movieId': df.index, 'labels': labels}).merge(df_movies, on='movieId').set_index('movieId')
    
    return df_labeledMovies