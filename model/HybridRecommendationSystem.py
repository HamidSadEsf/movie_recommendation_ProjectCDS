import pandas as pd
from model.ContentBasedRec import ContentBasedRecommender
from model.CollaborativeFilteringRec import CollaborativeFilteringRecommender as CFR, get_collaborative_filtering_weight
from model.ColdStarter import cold_starters

def get_user_movies_by_userid(userId):
    ratings = pd.read_csv('data/processed/final_ratings.csv')
    movies = ratings[ratings.userId == userId].movieId.values.tolist()
    return movies

class HybridRecommender():

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
    number_of_recommendations: Integer
            Number of desired recommendation, default is 20
    
    Returns
    ----------
    pandas.Dataframe
        The list of recommended movies and the corresponding score.
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
                the calculated hybrid score
        
    """
    
    def __init__(self):
        self.CBR = None
        self.cf_model = None
        
        
    def load_datasets(self):
        
        """
        loading dataset
        """
        self.CBR = ContentBasedRecommender()
        self.CBR.load_database()
        self.cf_model = CFR()
        self.cf_model.fit_and_predict()
        
        
    def hybrid_recommendation(self, userId, threshold=20, CFR2 = None, number_of_recommendations = 20):
        
        '''
        making recommendation
        '''
        
        # Getting the movies already rated by the user
        user_movies = get_user_movies_by_userid(userId)
        num_user_movies = len(user_movies)
        hybrid_rec = pd.DataFrame()
        
        # Case 1
        if num_user_movies < threshold:
            # Use content-based recommendation
            content_based_rec = self.CBR.recommendation(userId)
            
            coldstarter_rec = cold_starters(amount=0)
            # Calculate the weight for content-based recommendation
            
            content_based_weight = (num_user_movies / threshold)
            
            hybrid_rec = content_based_rec.merge(coldstarter_rec, on ='movieId', how= "outer").fillna(0)
            
            hybrid_rec['hybrid_score'] = (hybrid_rec['score'] * content_based_weight) + \
                            (hybrid_rec['cs_score'] * (1 - content_based_weight))
                            
            hybrid_rec = hybrid_rec[['movieId','hybrid_score']].merge(self.CBR.df_labeled.drop('labels', axis= 1), on='movieId')
                            
            
        # Case 2
        if num_user_movies >= threshold:
            # Use content-based recommendation
            content_based_rec = self.CBR.recommendation(userId)
            
            # Calculate the cf prediction
            if CFR2 is not None:
                self.cf_model = CFR2
            collaborative_filtering_rec = self.cf_model.get_rankings_for_movies(userId, content_based_rec.index.values)
            
            
            #Calculate the weight for collaborative filtering recommendation
            collaborative_filtering_weight = get_collaborative_filtering_weight(userId)
                                            
            # Apply weights to recommendations
            hybrid_rec = collaborative_filtering_rec.merge(content_based_rec, on='movieId')
            
            # We can try different cases here/
            hybrid_rec['hybrid_score'] = hybrid_rec['score'] + hybrid_rec['cf_score']*collaborative_filtering_weight
            
            #hybrid_rec_score = (hybrid_rec['score'] * (1 - collaborative_filtering_weight)) + (hybrid_rec['cf_score'] * collaborative_filtering_weight)

        hybrid_rec.sort_values(by='hybrid_score', ascending=False, inplace=True)
        
        if number_of_recommendations == 0:
            return hybrid_rec[["movieId", "title", "genres", "hybrid_score"]]
        else:
            return hybrid_rec[["movieId", "title", "genres", "hybrid_score"]].head(number_of_recommendations)

def get_HRMatrix():
    """
    getting the Matrix of users and movies with the predicted recommendation hybrid score by HR for each user and movie

    Returns:
        HR_Matrix: narray
        Matrix of dimension n_users and n_movies. Each value is the predicted hybrid score for each movie and user
    """
    hr = HybridRecommender()
    hr.load_datasets()
    final_rating = pd.read_csv(r'data\processed\final_ratings.csv')
    HRMatrix = pd.DataFrame(index = hr.CBR.df.index).sort_index()
    for userid in final_rating['userId'].unique()[:1000]:
        hr_score = hr.hybrid_recommendation(userid, number_of_recommendations=0).set_index('movieId').sort_index()
        HRMatrix= pd.concat([HRMatrix, hr_score.hybrid_score.rename(userid)], axis=1)
        print('user nr.', userid)
    HRMatrix = HRMatrix.transpose().rename_axis("userId", axis=0).rename_axis("movieId", axis=1)
    HRMatrix.to_csv('./data/processed/HRMatrix.csv', index_label="userId")

    return HRMatrix