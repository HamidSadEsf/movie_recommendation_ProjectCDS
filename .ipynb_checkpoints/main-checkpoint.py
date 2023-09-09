# Preprocess and save data for collaborative filtering modelling
# Uncomment only when you want to (re)build your ratings matrix from scratch

#from data_script.preprocess_collaborative import hamid_user_id, sveta_user_id

sveta_user_id = 78856
hamid_user_id = 70937

import numpy as np
import pandas as pd
from collections import defaultdict

from surprise import Dataset
from surprise import Reader
from surprise import BaselineOnly, Dataset, Reader, SVD, KNNBasic, KNNBaseline, KNNWithMeans

from surprise.model_selection import GridSearchCV
from surprise.model_selection import train_test_split

from model.CollaborativeFilteringRec import cf_model

#Get the final ratings matrix
print("Getting the ratings matrix...")
ratings = pd.read_csv('./data/processed/final_ratings.csv')
movie_df = pd.read_csv('./data/external/movies.csv')

#Prepare data in the Surprise's format
print("Preparing data in the Suprise format...")
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)

#trainset, testset = train_test_split(data, test_size=.25, random_state=42)

svd = SVD()

trainset = data.build_full_trainset()
testset = trainset.build_anti_testset()

svd_model = cf_model(svd, trainset, testset, data)
svd_model.fit_and_predict()
#svd_model.cross_validate()


print(svd_model.recommend(sveta_user_id, 20).merge(movie_df, on='movieId', how='inner'))
print(svd_model.recommend(hamid_user_id, 20).merge(movie_df, on='movieId', how='inner'))