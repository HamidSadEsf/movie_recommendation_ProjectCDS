import numpy as np
import pandas as pd
import os.path
import sklearn.neighbors
from sklearn.cluster import KMeans
import pickle

def get_user_movies_by_userid(userId, top):
    """
    getting a list of movies based on the userId
    
    Parameters
    ----------
    userId Integer
        ID of the desired User
    top Integer
        quantity of the recent rated movies to be return
    
    Returns
    ----------
    movies: list of integer
        The most recent rated movies by the user. the amount is defined by the argument top
    allmovies: list of integer
        All of the movies rated by the user
         
    """
    ratings = pd.read_csv('data/processed/final_ratings.csv')
    users_movie = ratings[ratings.userId == userId]
    recent_movies = users_movie.sort_values(by='timestamp')[:top]
    movies = recent_movies.movieId.values.tolist()
    allmovies = users_movie.movieId.values.tolist()
    return movies, allmovies

    
def train_nearest_neighbors_model(df=None):
    """
    function to train nearest neighbors model
    used only within the class
    """
    if df is None:
        df = pd.read_csv('./data/processed/df_ContBaseRec.csv')
        df.set_index('movieId', inplace=True)
    nn = sklearn.neighbors.NearestNeighbors(n_neighbors=df.shape[0])
    nn.fit(df.values)
    #save model to disk
    filename = 'model/trained_models/CB_nearest_neighbors.sav'
    pickle.dump(nn, open(filename, 'wb'))
    return nn

class ContentBasedRecommender():
    """
    Content Based recommender Class
    Making a recommendation by content-based Filtering:
    Our dataset is based on the gnome-scores.csv dataset of MovieLens:
    
        Step 1: Preprocessing the gnome-scores.csv to a dataset 
        Step 2: applying a preclustering to the dataset and get the labels for each movie. 
        Step 3: Calculate the mean vector of the given movies 
        Step 4: Rank the movies by their cosine similarity to the mean point with sklearn.metrics.pairwise.cosine_similarity
                The similarity score is calculated upon the relevance of the tags (see: data_script/Preprocess_Content_Based)
        Step 5: Calculate the distances of the movies to the mean vector with sklearn.neighbors.
        Step 6: Diversify the ranking of step 4 by subtracting the weighted (lambda) distances
        Step 7: removing the given movies and movies which are not in the same clusters as the given movies
        Step 8: Returning the desired number of top ranked movies 
    
    Parameters
    ----------
    given_movies:  List of Integers or List of Lists of two strings.
        If List of Integers: List of movieIds
            [movieId_1, movieId_n]
        If List of Lists of two strings: List of movies 
            [['movie Title_1','release year_2'],...,['movie Title_n','release year_n']]]
            This list will be processed within the function to a list of movieIds
    self.number_of_recommendations: Integer
        Number of desired recommendation
    num_clusters: Integer
        number of clusters for preclustering
    lambda_val: Float between 0-1
        The weight for diversification. (Default: Random value between 0-1 )

    Returns
    ----------
    pandas.Dataframe
        The list of recommended movies and their score.
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
    def __init__(self):
        self.df = None
        self.df_labeled = None
        self.df_recommendations = None
        
        path = './data/processed/CBMatrix.csv'
        if os.path.isfile(path) == True:
            self.df_recommendations = pd.read_csv(path)
           
        self.load_database()
    
    def load_database(self):
        
        """
        loading dataset
        """
                
        # Get the database
        #from data_script.Preprocess_Content_Based import get_df
        self.df = pd.read_csv('./data/processed/df_ContBaseRec.csv')
        self.df.set_index('movieId', inplace=True)
        
        # Clustering the data and getting the ladled movies dataset
        #from data_script.Clustering import get_labeledMovies
        #self.df_labeled = get_labeledMovies(self.df, num_clusters)
        self.df_labeled = pd.read_csv('./data/processed/df_labeledMovies.csv')
        self.df_labeled.set_index('movieId', inplace=True)
      
    def recommendation(self, userId, number_of_recommendations = 0, lambda_val = np.random.rand(), recompute = False):
        """
        Predict function
        """
        
        if recompute == False:
            tmpdf = pd.DataFrame(self.df_recommendations.set_index('userId').loc[userId]).dropna().sort_values(userId,ascending=False)
            if number_of_recommendations != 0:
                tmpdf = tmpdf[:number_of_recommendations]
            tmpdf = tmpdf.reset_index().rename(columns = {'index' : 'movieId', userId : 'score'})
            tmpdf['movieId'] = tmpdf['movieId'].astype(int)
            df = tmpdf.merge(self.df_labeled.drop('labels', axis= 1), on='movieId')
            return df
        else:
            # get at 25 movies is user rated more than 25. otherwise all the movies rated by user
            def get_movies(userId, top):
                # get the movies by userId
                given_movies_ids, all_movies = get_user_movies_by_userid(userId, top)
                numb_movies = len(given_movies_ids)
                if numb_movies < 25:
                    # Get rid of the movies not represented in the preprocessed database
                    given_movies_ids = [item for item in given_movies_ids if item in self.df.index.tolist()]

                else:
                    given_movies_ids = [item for item in given_movies_ids if item in self.df.index.tolist()]
                    if len(given_movies_ids) < 25:
                        get_movies(userId, top + (25 - numb_movies))
                return given_movies_ids, all_movies

            # get the recent rated movies and all the rated movies by the user
            given_movies_ids, all_movies = get_movies(userId, 25)

            # Calculating the mean point (vector) of the given movies
            mean_point = self.df.loc[given_movies_ids].mean()


            ## movie recommendation with cosine similarity

            from sklearn.metrics.pairwise import cosine_similarity
            # Get the relevance vector of the given movie
            given_movie_vector = mean_point.values.reshape(1, -1)
            # Calculate cosine similarity between the given movie and all other movies

            similarity_scores = cosine_similarity(self.df.values, given_movie_vector)


            ## Calculate Distances of mean vector to the nearest neighbors

            # Import and fit the model
            #nn = sklearn.neighbors.NearestNeighbors(n_neighbors=self.df.shape[0])
            #nn.fit(self.df)
            nn = pickle.load(open('model/trained_models/CB_nearest_neighbors.sav', 'rb'))

            # Getting the distances of any point in the self.df to mean point
            distances, indices = nn.kneighbors(mean_point.values.reshape(1, -1))
            # Normalize the data
            distances = (distances - distances.min()) / (distances.max() - distances.min())
            # Sorting the distance according to the original index of the self.df
            distances = distances[0][np.argsort(indices[0])]

            # Diversification
            score = pd.Series((similarity_scores.reshape(1, -1) - (lambda_val/4 * distances))[0], name="score")

            # Concatenate scores to the self.df
            df_scored = pd.concat([self.df_labeled.reset_index(names='movieId'), score.reset_index(drop=True)],
                                axis=1).set_index('movieId')

            # Remove given movie from the list of recommendation
            df_scored = df_scored.drop(all_movies)
            # Removing movies which are not in the same clusters as the given movies
            given_movies_clusters = self.df_labeled.loc[given_movies_ids, "labels"].unique()
            df_scored = df_scored[df_scored["labels"].isin(given_movies_clusters)]

            # getting the movies from the ladled set which shows the title and label
            if number_of_recommendations != 0:
                recommendations = df_scored.sort_values('score', ascending=False).head(number_of_recommendations).reset_index()
            else:
                #return all the recommendations
                recommendations = df_scored.sort_values('score', ascending=False).reset_index()
            return recommendations
    
def get_CBMatrix():
    """
    getting the Matrix of users and movies with the predicted recommendation score by CB for each user and movie

    Returns:
        CB_Matrix: narray
        Matrix of dimension n_users and n_movies. Each value is the predicted score for each movie and user
    """
    CBR = ContentBasedRecommender()
    CBR.load_database()
    final_rating = pd.read_csv('./data/processed/final_ratings.csv')
    CBMatrix = pd.DataFrame(index = CBR.df.index).sort_index()
    for userid in final_rating['userId'].unique():
        cbr = CBR.recommendation(userid).set_index('movieId').sort_index()
        CBMatrix[userid]= cbr.score
    CBMatrix = CBMatrix.transpose().rename_axis("userId", axis=0).rename_axis("movieId", axis=1)
    CBMatrix.to_csv('./data/processed/CBMatrix.csv', index_label="userId")

    return CBMatrix