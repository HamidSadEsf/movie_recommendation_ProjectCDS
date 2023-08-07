import pandas as pd
from model.ContentBasedRec import recommendation
from model.CollaborativeFilteringRec import get_predictions
from model.ColdStarter import cold_starters


# Hybrid recommender system
def hybrid_recommendation(userId, user_movies, threshold=20):
    num_user_movies = len(user_movies)
    hybrid_rec = pd.DataFrame()
    if num_user_movies < threshold:
        # Use content-based recommendation
        content_based_rec = recommendation(user_movies, 10, 18)
        coldstarter_rec = cold_starters()
        # Calculate the weight for content-based recommendation
        content_based_weight = (num_user_movies / threshold)
        hybrid_rec_score = (content_based_rec['score'] * content_based_weight) + \
                           (coldstarter_rec['score'] * 1 - content_based_weight)

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
        collaborative_filtering_weight = (collaborative_filtering_weight - collaborative_filtering_weight.min()) / 
                                         (collaborative_filtering_weight.max() - collaborative_filtering_weight.min())
        # Apply weights to recommendations
        hybrid_rec_score = (content_based_rec['score'] * 1- collaborative_filtering_weight) + \
                           (collaborative_filtering_rec['score'] * collaborative_filtering_weight)

    hybrid_rec = pd.DataFrame({'movieId': content_based_rec['movieId'], 'hybrid_score': hybrid_rec_score})
    hybrid_rec.sort_values(by='hybrid_score', ascending=False, inplace=True)

    return hybrid_rec
