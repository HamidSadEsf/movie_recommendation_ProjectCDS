import numpy as np
import pandas as pd
from collections import defaultdict
from surprise import accuracy
from surprise import Dataset
from surprise import Reader
from surprise import BaselineOnly, Dataset, Reader, SVD, KNNBasic, KNNBaseline, KNNWithMeans
from surprise.model_selection import cross_validate
from surprise.model_selection import GridSearchCV
from surprise.model_selection import train_test_split

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

print("Train the KnnWithMeans algo...")
sim_options = {'name': 'cosine',
               'min_support': 4,
               'user_based': False}

knn_withmeans = KNNWithMeans(k=25,sim_options=sim_options)
knn_withmeans.fit(trainset)
knn_withmeans_preds = knn_withmeans.test(testset)
print("KnnWithMeans's accuracy on the test data:",accuracy.rmse(knn_withmeans_preds))
print("KnnWithMeans's accuracy on the validation data:",accuracy.rmse(knn_withmeans_preds))