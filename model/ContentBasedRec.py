import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def recommendation(given_movies, number_of_recommendations, num_clusters, lambda_val=np.random.rand()):
    # get the database
    from data_script.Preprocess_Content_Based import get_df
    df = get_df()
    # clustering the data
    from data_script.Clustering import get_labeledMovies
    df_labeled = get_labeledMovies(df, num_clusters)
    # checking if given_movies is an item of class list
    if isinstance(given_movies, list):
        # checking if given_movies is a list of lists
        if all(isinstance(item, list) for item in given_movies):
        # checking if it fits the [['movie Title_1','release year_1'],...,['movie Title_n','release year_n']] format.
            if all(len(item) == 2 and all(isinstance(sub_item, str) for sub_item in item) for item in given_movies):
                # convert the movies lists to list of integers cotnaining only Movie_Ids
                from features.search_movieId import search_movieId
                given_movies_ids = search_movieId(given_movies)
            else: 
                raise ValueError("Invalid format for given_movies. If it's a list of lists, it should contain lists of two strings [['movie_Title, release_year]].")
        # checking if the list consists only of integers
        elif all(isinstance(item, int) for item in given_movies):
            given_movies_ids = given_movies
        else: 
            raise ValueError("Invalid format for given_movies. If it's a list, it should contain lists or integers.")
    else:
            raise ValueError("Invalid format for given_movies. It should be a list.")
    
    # get rid of the moveis not represented in the preprocessed database
    given_movies_ids = [item for item in given_movies_ids if item in df.index.tolist()]
    
    # calculating the mean point (vector) of the given movies
    mean_point = df.loc[given_movies_ids].mean()

    # movie recommendation with cousin similarity
    from sklearn.metrics.pairwise import cosine_similarity
    # Get the relevance vector of the given movie
    given_movie_vector = mean_point.values.reshape(1, -1)
    # Calculate cosine similarity between the given movie and all other movies
    similarity_scores = cosine_similarity(df.values, given_movie_vector)
    

    # distances as relevance score with nearest neighbors distances
    import sklearn.neighbors
    nn = sklearn.neighbors.NearestNeighbors(n_neighbors=df.shape[0])
    nn.fit(df)
    # getting the distances of any point in the df to mean point
    distances, indices = nn.kneighbors(mean_point.values.reshape(1, -1))
    # normalize the data between 0 and 1
    distances = (distances - distances.min()) / (distances.max() - distances.min())
    # Sort the distance according to the original index of the df
    relevance_score = distances[0][np.argsort(indices[0])]
    # calculate the relevance_score
    relevance_score = (1 - relevance_score)
    
    # Diversification
    score = pd.Series((similarity_scores.reshape(1, -1) - (lambda_val/4 * relevance_score))[0], name="score")

    df_scored = pd.concat([df_labeled.reset_index(names='movieId'), score.reset_index(drop=True)],
                          axis=1).set_index('movieId')

    # Remove the given movie from the list of recommendation
    df_scored = df_scored.drop(given_movies_ids)
    # deleting the movies which are not in the same clusters as the given movies
    given_movies_clusters = df_labeled.loc[given_movies_ids, "labels"].unique()
    df_scored = df_scored[df_scored["labels"].isin(given_movies_clusters)]

    # getting the movies from the labled set which shows the title and label
    recommendations = df_scored.sort_values('score', ascending=False).head(number_of_recommendations)
    return recommendations

