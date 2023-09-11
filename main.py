# preprocess and save data for collaborative filtering modelling
#from data_script.preprocess_collaborative import hamid_user_id, sveta_user_id, anti_hamid_user_id, anti_sveta_user_id

hamid_user_id, sveta_user_id, anti_hamid_user_id, anti_sveta_user_id = (88764,71285,5737,58746)

# training and saving CF models
from model.CollaborativeFilteringRec import train_cf_models, CollaborativeFilteringRecommender, get_collaborative_filtering_weight
#trainset, testset, data, algo, algo_predictions, knn, knn_predictions = train_cf_models()

# Load the trained models
#from surprise import dump
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
from model.HybridRecommendationSystem import hybrid_recommendation
hybrid_recommendation(sveta_user_id)


