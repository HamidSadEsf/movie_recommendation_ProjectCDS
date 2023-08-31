import numpy as np
import pandas as pd
from surprise import accuracy
from surprise import Dataset
from surprise import Reader
from surprise import BaselineOnly, Dataset, Reader, KNNBasic
from surprise.model_selection import cross_validate
from surprise.model_selection import GridSearchCV
from surprise.model_selection import train_test_split

from model.collaborative_filtering_models.CollaborativeFilteringModel import collab_filtering_based_recommender_model

#Get the final ratings matrix
print("Getting the ratings matrix...")
ratings = pd.read_csv('./data/processed/final_ratings.csv')
movie_df = pd.read_csv('./data/external/movies.csv')

#Prepare data in the Surprise's format
print("Preparing data in the Suprise format...")
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)

#Prepare train and test data (what about validation?)
trainset, testset = train_test_split(data, test_size=.25, random_state=42)


#KNN-based memory based model
print("Train the KnnBasic algo...")
sim_options = {'name': 'msd',
               'min_support': 5,
               'user_based': False}

knn_basic = KNNBasic(k=30,sim_options=sim_options)
knn_basic.fit(trainset)
knn_basic_preds = knn_basic.test(testset)
print("KNNBasic's accuracy on the test data:",accuracy.rmse(knn_basic_preds))
print("KNNBasic's accuracy on the validation data:",accuracy.rmse(knn_basic_preds))