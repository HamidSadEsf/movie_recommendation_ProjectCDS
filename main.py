import pandas as pd
import numpy as np
from surprise import dump
import pickle
from evaluation.Evaluate import Evaluate, Evaluate_CFR
from model.CollaborativeFilteringRec import CollaborativeFilteringRecommender, train_cf_models, get_collaborative_filtering_weights
from model.HybridRecommendationSystem import HybridRecommender, get_HRMatrix
import model.ContentBasedRec as CBR

ratings = pd.read_csv('./data/processed/final_ratings.csv')
random_user = np.random.choice(ratings.userId.unique()) # picks a random user

# preprocess and save data for collaborative filtering modelling
#from data_script.preprocess_collaborative import hamid_user_id, sveta_user_id, anti_hamid_user_id, anti_sveta_user_id

# training and saving CF models
#trainset, testset, data, algo, algo_predictions, knn, knn_predictions = train_cf_models()

# Load the trained models
#cf_predictions, cf_recommender_algo = dump.load('./model/trained_models/CF_Model')
#CFR = CollaborativeFilteringRecommender(cf_predictions, cf_recommender_algo)
#CFR.fit_and_predict() # performs all the computation
#CFR.cross_validate()
#CFR = CollaborativeFilteringRecommender()
#print(len(CFR.recommenddf.userId.unique()))
#print(len(CFR.recommenddf.movieId.unique()))
# Get top n recommendations for a user
#recommendations = CFR.recommend(sveta_user_id, n=10)
#print(recommendations.head(20))

# Get CF weight coefficients for the hybrid model
#print(get_collaborative_filtering_weight(anti_sveta_user_id))

# Test content-based recommender
#df = pd.read_csv('./data/processed/df_ContBaseRec.csv')
#df.set_index('movieId', inplace=True)
#nn = CBR.train_nearest_neighbors_model(df)
#cb_df = CBR.recommendation(62095, 20)
#cb_df.head(20)

# Test hybrid recommendations
#hyb_recommender = HybridRecommender()
#userId = 186
#print(hyb_recommender.hybrid_recommendation(userId, 20, recompute = True))

# Test evaluation

#cf_predictions_preds, cf_algo = dump.load('./model/trained_models/CF_Model')
#cf_model = CFR(cf_predictions_preds, cf_algo)
#cf_model.fit_and_predict()

#eval_cf_model = Evaluate_CFR(cf_model)
#print(eval_cf_model.variety())

evaluate = Evaluate()
evaluate.compute_surprise_similarity()

#Personalization
result_cf = evaluate.personalisation('cf')
result_cb = evaluate.personalisation('cb')

#Variety

#print(evaluate.variety_collaborative_filtering())
#print(evaluate.variety_content_based())
#print(evaluate.variety_hybrid())

#from model.ContentBasedRec import recommendation
#from model.HybridRecommendationSystem import hybrid_recommendation
#ratings = pd.read_csv('./data/processed/final_ratings.csv')
#random_user = np.random.choice(ratings.userId.unique())
#df = recommendation(random_user, 20, 0)
#df = hybrid_recommendation(54124)
#print(df.head(20))

#Testing new CB classes:
#from model.ContentBasedRec import ContentBasedRecommender
#cb_recommender = ContentBasedRecommender()
#print(cb_recommender.df_recommendations.head())
