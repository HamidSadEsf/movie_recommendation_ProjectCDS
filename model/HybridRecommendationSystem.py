import pandas as pd
from model.ContentBasedRec import recommendation
from model.CollaborativeFilteringRec import CollaborativeFilteringRecommender as CFR, get_collaborative_filtering_weight
from model.ColdStarter import cold_starters

def get_user_movies_by_userid(userId):
    ratings = pd.read_csv('data/processed/final_ratings.csv')
    movies = ratings[ratings.userId == userId].movieId.values.tolist()
    return movies

def hybrid_recommendation(userId, threshold=20):
    """
    Making a hybrid recommendation list from the three recommendation Systems:
    Cold starter (CS), Content-Based (CB), Collaborative Filtering (CF)
    Case 1:
        If the user rated fewer movies as indicated in the threshold, it returns a
        hybrid recommendation of CB and CS.
        The weighting depends on the number of movies rated by the user.
        The more movies the user rates, the more weight is given to the CB.
        if the amount of movies rated is zero, then it returns only CS
    Case 2:
        If the user rated more movies as indicated in the threshold, it returns a
        hybrid recommendation of CB and CF.
        The weighting depends on the number of the 'similar" users as determined from the similarity matrix.
    
    Parameters
    ----------
    UserId:  integer
        The Id of the user to whom we want to recommend movies. 
    threshold: integer
        The number of movies rated by the user as the threshold for Case 1 and Case 2
    
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
    # Getting the movies already rated by the user
    user_movies = get_user_movies_by_userid(userId)
    num_user_movies = len(user_movies)
    hybrid_rec = pd.DataFrame()
    
    # Case 1
    if num_user_movies < threshold:
        # Use content-based recommendation
        content_based_rec = recommendation(userId, 20)
        
        coldstarter_rec = cold_starters()
        # Calculate the weight for content-based recommendation
        
        content_based_weight = (num_user_movies / threshold)
        hybrid_rec_score = (content_based_rec['score'] * content_based_weight) + \
                           (coldstarter_rec['score'] * 1 - content_based_weight)

    # Case 2
    if num_user_movies >= threshold:
        # Use content-based recommendation
        content_based_rec = recommendation(userId, 0)
        
        # Calculate the cf prediction
        cf_model = CFR()
        cf_model.fit_and_predict()
        collaborative_filtering_rec_1 = cf_model.recommend(userId, 20)
        collaborative_filtering_rec = cf_model.get_rankings_for_movies(userId, content_based_rec.index.values)
        
        
        #Calculate the weight for collaborative filtering recommendation
        collaborative_filtering_weight = get_collaborative_filtering_weight(userId)
                                         
        # Apply weights to recommendations
        hybrid_rec = collaborative_filtering_rec.merge(content_based_rec, on='movieId')
        #hybrid_rec_score = (hybrid_rec['score'] * (1 - collaborative_filtering_weight)) + (hybrid_rec['cf_score'] * collaborative_filtering_weight)
        
        # We can try different cases here/
        hybrid_rec_score = hybrid_rec['score'] + hybrid_rec['cf_score']*collaborative_filtering_weight
        hybrid_rec['hybrid_score'] = hybrid_rec_score

    #hybrid_rec = pd.DataFrame({'movieId': content_based_rec['movieId'], 'title': content_based_rec['title'], 'genre': content_based_rec['genres'], hybrid_score: 'hybrid_rec_score'})
    hybrid_rec.sort_values(by='hybrid_score', ascending=False, inplace=True)
    return hybrid_rec[["movieId", "title", "genres", "hybrid_score"]]