import numpy as np
import pandas as pd
from surprise import dump
import random
from model.CollaborativeFilteringRec import CollaborativeFilteringRecommender
import model.HybridRecommendationSystem as HR
import model.ContentBasedRec as CBR

class Evaluate_CFR():
    def __init__(self, model):
        self.model = model

    def variety(self, number_of_users = 100, number_of_recommendations = 10):
        var = []
        unique_movies = set()
        for i in range(10):
            # Select users randomly
            all_users = self.model.recommenddf.userId.unique().tolist()
            selection_of_inner_users = np.array(random.sample(all_users, number_of_users))
            for u in selection_of_inner_users:
                # While computing 10 recommendatiosn for each, count only unique movies
                recommendations = self.model.recommend(u, number_of_recommendations)
                movie_list = set(recommendations.movieId.values.tolist())
                unique_movies = unique_movies | movie_list
            
            var.append(len(sorted(unique_movies)))
        return round(0.001*np.array(var).mean()-0.01,3), round(0.001*np.array(var).std(),3)
    
    def Coverage(self):
        return len(self.model.pred_test) / len(self.testset)

    
    
class Evaluate():
    def __init__(self):
        cf_predictions, cf_recommender_algo = dump.load('./model/trained_models/CF_Model')
        CFR = CollaborativeFilteringRecommender(cf_predictions, cf_recommender_algo)
        CFR.fit_and_predict()
        self.cf_model = CFR
        self.cb_model = CBR
        self.hybrid_model = HR
        self.variety_nb_users = 100
        self.variety_nb_recommendations = 10
        self.var_repetitions = 1
        ratings = pd.read_csv('./data/processed/final_ratings.csv')
        self.userIds = ratings.userId.unique().tolist()

    def variety_collaborative_filtering(self):
        var = []
        unique_movies = set()
        for i in range(self.var_repetitions):
            # Select users randomly
            selection_of_users = np.array(random.sample(self.userIds, self.variety_nb_users))
            for u in selection_of_users:
                # While computing 10 recommendatiosn for each, count only unique movies
                recommendations = self.cf_model.recommend(u, self.variety_nb_recommendations)
                movie_list = set(recommendations.movieId.values.tolist())
                unique_movies = unique_movies | movie_list
            
            var.append(len(sorted(unique_movies)))
        return round(0.001*np.array(var).mean()-0.01,3), round(0.001*np.array(var).std(),3)
    

    def variety_content_based(self):
        variety = []
        unique_movies = set()
        for i in range(self.var_repetitions):
            # Select users randomly
            selection_of_users =  np.array(random.sample(self.userIds, self.variety_nb_users))
            for u in selection_of_users:
                # While computing 10 recommendatiosn for each, count only unique movies
                recommendations = self.cb_model.recommendation(u, self.variety_nb_recommendations)
                movie_list = set(recommendations.movieId.values.tolist())
                unique_movies = unique_movies | movie_list
            
            variety.append(len(sorted(unique_movies)))
        return round(0.001*np.array(variety).mean()-0.01,3), round(0.001*np.array(variety).std(),3)
    
    def variety_hybrid(self):
        variety = []
        unique_movies = set()
        for i in range(self.var_repetitions):
            # Select users randomly
            selection_of_users =  np.array(random.sample(self.userIds, self.variety_nb_users))
            for u in selection_of_users:
                # While computing 10 recommendatiosn for each, count only unique movies
                recommendations = self.hybrid_model.hybrid_recommendation(u, self.variety_nb_recommendations, self.cf_model)
                movie_list = set(recommendations.movieId.values.tolist())
                unique_movies = unique_movies | movie_list
            
            variety.append(len(sorted(unique_movies)))
        return round(0.001*np.array(variety).mean()-0.01,3), round(0.001*np.array(variety).std(),3)
    
    def Coverage(self):
        return len(self.model.pred_test) / len(self.testset)

    
    #def personalisation_cf(self):
        
