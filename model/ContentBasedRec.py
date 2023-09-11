import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def recommendation(given_movies, number_of_recommendations, num_clusters, lambda_val=np.random.rand()):
    
    """
    Making a recommendation by content-based Filtering:
    Our dataset is based on the gnome-scores.csv dataset of MovieLens:
    
        Step 1: Preprocessing the gnome-scores.csv to a dataset 
        Step 2: Apply a pre-clustering to the dataset and get the labels for each movie. 
        Step 3: Calculate the mean vector of the given movies 
        Step 4: Rank the movies by their cosine similarity to the mean point with sklearn.metrics.pairwise.cosine_similarity
                The similarity score is calculated upon the relevance of the tags (see: data_script/Preprocess_Content_Based)
        Step 5: Calculate the distances of the movies to the mean vector with sklearn.neighbors.
        Step 6: Diversify the ranking of step 4 by subtracting the weighted (lambda) distances
        Step 7: removing the given movies and movies which are not in the same clusters as the given movies
        Step 8: Returning the desired number of top-ranked movies 
    
    In case the number of recommendations is set to zero, it returns the whole dataframe with calculated score for every movie
    
    Parameters
    ----------
    given_movies:  List of Integers or List of Lists of two strings.
        If List of Integers: List of movieIds
            [movieId_1, movieId_n]
        If List of Lists of two strings: List of movies 
            [['movie Title_1','release year_2'],...,['movie Title_n','release year_n']]]
            This list will be processed within the function to a list of movieIds
    number_of_recommendations: Integer
        Number of desired recommendation
    num_clusters: Integer
        number of clusters for pre-clustering
    lambda_val: Float between 0-1
        The weight for diversification. (Default: Random value between 0-1 )

    Returns
    ----------
    pandas.Dataframe
        The top rows of a sorted data frame of recommended movies, sorted by their score. The number of rows is specified by number_of_recommendations
        If the number of recommendations is set to zero, then an unsorted and uncut data frame is returned.
        Index: movieId
        Columns:
            labels:
                integer
                The cluster of the movie
            title:
                String
                title(release year)
            genres:
                String 
                genres1|genre2|...|genren
            score
                Float
                The calculated score between 0-1
        
    """
    
    # Get the database
    from data_script.Preprocess_Content_Based import get_df
    df = get_df()
    
    # Clustering the data and getting the labeled movies dataset
    from data_script.Clustering import get_labeledMovies
    df_labeled = get_labeledMovies(df, num_clusters)
    
    
    ## Checking if the given movies already contain the Ids. if not get the Id
    
    # Checking if given_movies is an item of the class list
    if isinstance(given_movies, list):
        # Checking if given_movies is a list of lists
        if all(isinstance(item, list) for item in given_movies):
        # Checking if it fits the [['movie Title_1','release year_1'],...,['movie Title_n','release year_n']] format.
            if all(len(item) == 2 and all(isinstance(sub_item, str) for sub_item in item) for item in given_movies):
                # Convert the movie lists to a list of integers containing only Movie_Ids
                from features.search_movieId import search_movieId
                given_movies_ids = search_movieId(given_movies)
            else: 
                raise ValueError("Invalid format for given_movies. If it's a list of lists, it should contain lists of two strings [['movie_Title, release_year]].")
        # Checking if the list consists only of integers
        elif all(isinstance(item, int) for item in given_movies):
            given_movies_ids = given_movies
        else: 
            raise ValueError("Invalid format for given_movies. If it's a list, it should contain lists or integers.")
    else:
            raise ValueError("Invalid format for given_movies. It should be a list.")
    
    # Get rid of the movies not represented in the preprocessed database
    given_movies_ids = [item for item in given_movies_ids if item in df.index.tolist()]
    
    # Calculating the mean point (vector) of the given movies
    mean_point = df.loc[given_movies_ids].mean()


    ## Movie recommendation with cousin similarity
    
    from sklearn.metrics.pairwise import cosine_similarity
    # Get the relevance vector of the given movie
    given_movie_vector = mean_point.values.reshape(1, -1)
    # Calculate cosine similarity between the given movie and all other movies
    similarity_scores = cosine_similarity(df.values, given_movie_vector)
    

    ## Calculate Distances of the mean vector to the nearest neighbors
    
    # Import and fit the model
    import sklearn.neighbors
    nn = sklearn.neighbors.NearestNeighbors(n_neighbors=df.shape[0])
    nn.fit(df)
    # Getting the distances of any point in the df to mean point
    distances, indices = nn.kneighbors(mean_point.values.reshape(1, -1))
    # Normalize the data
    distances = (distances - distances.min()) / (distances.max() - distances.min())
    # Sorting the distance according to the original index of the df
    distances = distances[0][np.argsort(indices[0])]
    
    # Diversification
    score = pd.Series((similarity_scores.reshape(1, -1) - (lambda_val/4 * distances))[0], name="score")

    # Concatenate scores to the df
    df_scored = pd.concat([df_labeled.reset_index(names='movieId'), score.reset_index(drop=True)],
                          axis=1).set_index('movieId')

    # Remove given movies from the list of recommendation
    df_scored = df_scored.drop(given_movies_ids)
    # Removing movies that are not in the same clusters as the given movies
    given_movies_clusters = df_labeled.loc[given_movies_ids, "labels"].unique()
    df_scored = df_scored[df_scored["labels"].isin(given_movies_clusters)]
    #if the number of recommendation is set to zero, we get the score data frame. 
    if number_of_recommendations == 0:
        return df_scored
    #otherwise we get the ranking
    else:
        # getting the movies from the labeled set which shows the title and label
        recommendations = df_scored.sort_values('score', ascending=False).head(number_of_recommendations)
        
        return recommendations

