import numpy as np
import pandas as pd
from surprise import dump
import random
import matplotlib.pyplot as plt
from surprise import SVD, KNNBasic, KNNWithMeans

from model.CollaborativeFilteringRec import CollaborativeFilteringRecommender, prepare_data
from model.HybridRecommendationSystem import HybridRecommender
from model.ContentBasedRec import ContentBasedRecommender

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
        print('...Loading Content Based Recommender..')
        self.cbr = ContentBasedRecommender()
        print('...Loading Hybrid Recommender..')
        self.hr = HybridRecommender()
        print('...Loading Collaborative Filtering Recommender..')
        self.cf_model = CollaborativeFilteringRecommender()
        self.cf_model.recompute_surprise_data() # sets trainset etc
        self.variety_nb_users = 100
        self.variety_nb_recommendations = 10
        self.var_repetitions = 10
        ratings = pd.read_csv('./data/processed/final_ratings.csv')
        self.userIds = ratings.userId.unique().tolist()
        self.knn_similarities = None
     
    def compute_surprise_similarity(self):
        print('...Computing KNN similarity matrix...')
        sim_options = {'name': 'msd',
               'min_support': 3,
               'user_based': True}

        knn = KNNBasic(k=25,sim_options=sim_options)
        knn.fit(self.cf_model.trainset)
        #knn_predictions = knn.test(self.testset)
        self.knn_similarities = knn.compute_similarities()
        return

    def compute_prediction_overlap(self, userId_1, userId_2, model_type = 'cf'):
        overlap_array = []
        if model_type == 'cf':
            rec_1 = set(self.cf_model.recommend(userId_1, 20).movieId.values)
            rec_2 = set(self.cf_model.recommend(userId_2, 20).movieId.values)
        elif model_type == 'cb':
            rec_1 = set(self.cbr.recommendation(userId_1, 20).movieId.values)
            rec_2 = set(self.cbr.recommendation(userId_2, 20).movieId.values)
        elif model_type == 'hyb':
            rec_1 = set(self.hr.hybrid_recommendation(userId_1, 20).movieId.values)
            rec_2 = set(self.hr.hybrid_recommendation(userId_2, 20).movieId.values)
            
        overlap = rec_1 
        return len(rec_1.intersection(rec_2))
            
    def return_similarities(self, userId, model_type):
        df = pd.DataFrame(columns=["uuid", "similarity", "overlap"])
        uuid = self.cf_model.trainset.to_inner_uid(userId)
        sim_array = self.knn_similarities[uuid]
        uuid_array = np.arange(self.knn_similarities.shape[0])
        df["uuid"] = uuid_array
        df["similarity"] = sim_array
        df["overlap"] = df.apply(lambda x: self.compute_prediction_overlap(userId, self.cf_model.trainset.to_raw_uid(x["uuid"]), model_type), axis=1)
        return df.sort_values(by=["similarity"]).reset_index(drop=True)
    
    def personalisation(self, model_type):
        random_user = np.random.choice(self.userIds) # picks a random user
        df_perso = self.return_similarities(random_user, model_type)
        return df_perso

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
                recommendations = self.cbr.recommendation(u, self.variety_nb_recommendations)
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
                recommendations = self.hr.hybrid_recommendation(u, self.variety_nb_recommendations)
                movie_list = set(recommendations.movieId.values.tolist())
                unique_movies = unique_movies | movie_list
            
            variety.append(len(sorted(unique_movies)))
        return round(0.001*np.array(variety).mean()-0.01,3), round(0.001*np.array(variety).std(),3)
    
    def Coverage(self):
        return len(self.model.pred_test) / len(self.testset)

        
        
        
        
