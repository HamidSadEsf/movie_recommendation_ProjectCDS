import pandas as pd
import numpy as np
from surprise import dump
from evaluation.Evaluate import Evaluate
from model.CollaborativeFilteringRec import CollaborativeFilteringRecommender as CFR, train_cf_models, get_collaborative_filtering_weight

# preprocess and save data for collaborative filtering modelling
#from data_script.preprocess_collaborative import hamid_user_id, sveta_user_id, anti_hamid_user_id, anti_sveta_user_id

# training and saving CF models
#trainset, testset, data, algo, algo_predictions, knn, knn_predictions = train_cf_models()

# Load the trained models
#cf_predictions, cf_recommender_algo = dump.load('./model/trained_models/CF_Model')
#CFR = CF(cf_predictions, cf_recommender_algo)
#CFR.fit_and_predict() # performs all the computation
#CFR.cross_validate()

# Get top n recommendations for a user
#recommendations = CFR.recommend(sveta_user_id, n=10)
#print(recommendations.head(20))

# Get CF weight coefficients for the hybrid model
#print(get_collaborative_filtering_weight(anti_sveta_user_id))

# Test hybrid recommendations
#from model.HybridRecommendationSystem import hybrid_recommendation
#hybrid_recommendation(sveta_user_id)

# Test evaluation
#knn_wmeans_preds, knn_wmeans_algo = dump.load('./model/trained_models/CF_Model')
#knn_with_means_model = CFR(knn_wmeans_preds, knn_wmeans_algo)
#knn_with_means_model.fit_and_predict()
#trainset = knn_with_means_model.trainset

#eval_knn_wmeans = Evaluate(knn_with_means_model, trainset)
#print(eval_knn_wmeans.Variety())

#from model.ContentBasedRec import recommendation
from model.HybridRecommendationSystem import hybrid_recommendation
ratings = pd.read_csv('./data/processed/final_ratings.csv')
random_user = np.random.choice(ratings.userId.unique())
#df = recommendation(random_user, 20, 0)
df = hybrid_recommendation(68729)
print(df.head(20))