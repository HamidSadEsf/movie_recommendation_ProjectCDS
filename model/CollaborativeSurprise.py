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

def get_movie_titles_by_ids(ids, movie_df):
    titles = []
    for i in ids.astype(int):
        title = movie_df[movie_df.movieId == i].title
        titles.append(title)
    return titles

def get_top_n(predictions, n=10):
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

class collab_filtering_based_recommender_model():
    def __init__(self, model, trainset, testset, data):
        self.model = model
        self.trainset = trainset
        self.testset = testset
        self.data = data
        self.pred_test = None
        self.recommendations = None
        self.top_n = None
        self.recommenddf = None

    def fit_and_predict(self):        
        printmd('**Fitting the train data...**', color='brown')
        self.model.fit(self.trainset)       

        printmd('**Predicting the test data...**', color='brown')
        self.pred_test = self.model.test(self.testset)        
        rmse = round(accuracy.rmse(self.pred_test), 3)
        printmd('**RMSE for the predicted result is ' + str(rmse) + '**', color='brown')   
        
        self.top_n = get_top_n(self.pred_test)
        self.recommenddf = pd.DataFrame(columns=['userId', 'movieId', 'rating'])
        for item in self.top_n:
            subdf = pd.DataFrame(self.top_n[item], columns=['movieId', 'rating'])
            subdf['userId'] = item
            cols = subdf.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            subdf = subdf[cols]        
            self.recommenddf = pd.concat([self.recommenddf, subdf], axis = 0)
        return rmse
        
    def cross_validate(self):
        printmd('**Cross Validating the data...**', color='brown')
        cv_result = cross_validate(self.model, self.data, n_jobs=-1)
        cv_result = round(cv_result['test_rmse'].mean(),3)
        printmd('**Mean CV RMSE is ' + str(cv_result)  + '**', color='brown')
        return cv_result

    def recommend(self, user_id, n=5):
        printmd('**Recommending top ' + str(n)+ ' products for userid : ' + str(user_id) + ' ...**', color='brown')
        
        #df = pd.DataFrame(self.top_n[user_id], columns=['productId', 'Rating'])
        #df['UserId'] = user_id
        #cols = df.columns.tolist()
        #cols = cols[-1:] + cols[:-1]
        #df = df[cols].head(n)
        
        df = self.recommenddf[self.recommenddf['userId'] == user_id].head(n)
        display(df)
        return df

#Prepare train and test data (what about validation?)
trainset, testset = train_test_split(data, test_size=.25, random_state=42)

########################################
############### MODELS #################
########################################

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



print("Train the KnnWithMeans algo...")
sim_options = {'name': 'cosine',
               'min_support': 4,
               'user_based': False}

knn_withmeans = KNNWithMeans(k=25,sim_options=sim_options)
knn_withmeans.fit(trainset)
knn_withmeans_preds = knn_withmeans.test(testset)
print("KnnWithMeans's accuracy on the test data:",accuracy.rmse(knn_withmeans_preds))
print("KnnWithMeans's accuracy on the validation data:",accuracy.rmse(knn_withmeans_preds))

