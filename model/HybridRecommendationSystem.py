import pandas as pd
from model.ContentBasedRec import recommendation
from model.CollaborativeFilteringRec import get_predictions
from model.ColdStarter import cold_starters

def hybrid_recommendation(userId, threshold=20):
     """
    Making a hybrid recommendation list from the three recommendation Systems:
    Cold Starte (CS), Content Based (CB), Colaborative Filtering (CF)
    Case 1:
        If user rated less movies as indicated in threshold, it returns a
        hybrid recommendation of CB and CS.
        The weighting of depends on the amount of movies rated by the user.
        The more movies the user rates, the more weight is given to the CB.
        if amount of movies rated is zero, then it return only CS
    Case 2:
        If user rated more movies as indicated in threshold, it returns a
        hybrid recommendation of CB and CF.
        The weighting of depends on ....
    
    Parameters
    ----------
    UserId:  integer
        The Id of the user to whom we want to recommend movies. 
    threshold: integer
        The amount of movies rated by user as threshold for Case 1 and Case 2
    
    Returns
    ----------
    pandas.Dataframe
        The list of recommended movies and their score.
        rows: movies 
        Columns:
            movieId:
                integer
            title:
                String
                title(release year)
            genres:
                String 
                genres1|genre2|...|genren
            hybrid_score
                Float
                the calculated hybrid score between 0-1
        
    """
    # geting the movies already rated by user
    user_movies = = pd.read_csv('data/ratings.csv')['movieId']
    num_user_movies = len(user_movies)
    hybrid_rec = pd.DataFrame()
    
    # Case 1
    if num_user_movies < threshold:
        # Use content-based recommendation
        content_based_rec = recommendation(user_movies, 10, 18)
        coldstarter_rec = cold_starters()
        # Calculate the weight for content-based recommendation
        content_based_weight = (num_user_movies / threshold)
        hybrid_rec_score = (content_based_rec['score'] * content_based_weight) + \
                           (coldstarter_rec['score'] * 1 - content_based_weight)

    # Case 2
    if num_user_movies >= threshold:
        # Calculate the weight for collaborative filtering recommendation
        collaborative_filtering_rec = get_predictions(userId)
        #getting the mean of the actual rating
        actual_ratings = user_movies['rating'].mean()
        # getting the mean of the predicted rating
        predicted_ratings = collaborative_filtering_rec[(collaborative_filtering_rec.moviedId == user_movies['movie_id'])['score']].mean()
        #calculating the weight by taking the difference between actual and predicted ratings
        collaborative_filtering_weight = actual_ratings - predicted_ratings
        # normalizing the weight
        collaborative_filtering_weight = (collaborative_filtering_weight - collaborative_filtering_weight.min()) / (collaborative_filtering_weight.max() - collaborative_filtering_weight.min())
        # Apply weights to recommendations
        hybrid_rec_score = (content_based_rec['score'] * 1- collaborative_filtering_weight) + \
                           (collaborative_filtering_rec['score'] * collaborative_filtering_weight)

    hybrid_rec = pd.DataFrame({'movieId': content_based_rec['movieId'], 'title': content_based_rec['title'], 'genre': content_based_rec['genres'], hybrid_score': hybrid_rec_score'})
    hybrid_rec.sort_values(by='hybrid_score', ascending=False, inplace=True)

    return hybrid_rec
