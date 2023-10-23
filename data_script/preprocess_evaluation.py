#KNN-based memory based model
import pandas as pd
import numpy as np
from surprise import dump
from surprise import KNNBasic, KNNBaseline, KNNWithMeans, Dataset, Reader

def prepare_data():
    print("Getting the ratings matrix...")
    ratings = pd.read_csv('./data/processed/final_ratings.csv')
    print("Preparing data in the Suprise format...")
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
    #Prepare train and test data (what about validation?)
    #trainset, testset = train_test_split(data, test_size=.25, random_state=42)
    # For final preditions
    trainset = data.build_full_trainset()
    testset = trainset.build_anti_testset()
    return trainset, testset, data

trainset, testset, data = prepare_data()

sim_options = {'name': 'msd',
               'min_support': 3,
               'user_based': True}

knn_basic = KNNBasic(k=25,sim_options=sim_options)
knn_basic.fit(trainset)
knn_basic_predictions = knn_basic.test(testset)

knn_basic_model = CFR.CollaborativeFilteringRecommender(knn_basic_predictions, knn_basic, file_name = KNN)
knn_basic_model.fit_and_predict()