import numpy as np
import pandas as pd
import sklearn.neighbors
from sklearn.cluster import KMeans
import pickle

def get_user_movies_by_userid(userId):
    ratings = pd.read_csv('data/processed/final_ratings.csv')
    movies = ratings[ratings.userId == userId].movieId.values.tolist()
    return movies
    
def train_nearest_neighbors_model(df=None):
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
    """Content Based recommender Class"""
    
    def __inti__(self):
        self.df = None
        self.df_labeled = None
    
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
      
    def recommendation(self, userId, lambda_val = np.random.rand(), number_of_recommendations = 0):
        
        """
        Predict function
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
        given_movies = get_user_movies_by_userid(userId)
        
        ## Checking if the given movies already contain the Ids. if not get the Id
        
        # Checking if given_movies is an item of class list
        if isinstance(given_movies, list):
            # Checking if given_movies is a list of lists
            if all(isinstance(item, list) for item in given_movies):
            # Checking if it fits the [['movie Title_1','release year_1'],...,['movie Title_n','release year_n']] format.
                if all(len(item) == 2 and all(isinstance(sub_item, str) for sub_item in item) for item in given_movies):
                    # Convert the movies lists to list of integers containing only Movie_Ids
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
        given_movies_ids = [item for item in given_movies_ids if item in self.df.index.tolist()]
        
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
        df_scored = df_scored.drop(given_movies_ids)
        # Removing movies which are not in the same clusters as the given movies
        given_movies_clusters = self.df_labeled.loc[given_movies_ids, "labels"].unique()
        df_scored = df_scored[df_scored["labels"].isin(given_movies_clusters)]

        # getting the movies from the ladled set which shows the title and label
        if number_of_recommendations != 0:
            recommendations = df_scored.sort_values('score', ascending=False).head(self.number_of_recommendations).reset_index()
        else:
            #return all the recommendations
            recommendations = df_scored.sort_values('score', ascending=False).reset_index()
        return recommendations