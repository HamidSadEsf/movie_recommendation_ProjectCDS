import numpy as np
import pandas as pd
from collections import defaultdict
from surprise import accuracy
from surprise.model_selection import cross_validate

def get_top_n(predictions, n = 0):
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        if (n==0):
            top_n[uid] = user_ratings
        else:
            top_n[uid] = user_ratings[:n]
    return top_n

class cf_model():
    def __init__(self, model, trainset, testset, data, n=20, pred_test=None):
        self.model = model
        self.trainset = trainset
        self.testset = testset
        self.data = data
        self.pred_test = pred_test
        self.recommendations = None
        self.top_n = None
        self.recommenddf = None
        self.all_recommenddf = None
        self.all_predictions = None
        self.n = n 
        self.mean_cv_rmse = None
        self.similarity_matrix = None

    def fit_and_predict(self):
        if (self.pred_test is None):
            #print('Fitting the train data...')
            self.model.fit(self.trainset)       

            #print('Predicting the test data...')
            self.pred_test = self.model.test(self.testset)
            
        rmse = round(accuracy.rmse(self.pred_test), 3)
        #print('RMSE for the predicted result is ' + str(rmse))   
        
        self.top_n = get_top_n(self.pred_test, self.n)
        self.all_predictions = get_top_n(self.pred_test)
        self.recommenddf = pd.DataFrame(columns=['userId', 'movieId', 'pred_rating'])
        self.all_recommenddf = pd.DataFrame(columns=['userId', 'movieId', 'pred_rating'])
        for item in self.top_n:
            subdf = pd.DataFrame(self.top_n[item], columns=['movieId', 'pred_rating'])
            subdf['userId'] = item
            cols = subdf.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            subdf = subdf[cols]        
            self.recommenddf = pd.concat([self.recommenddf, subdf], axis = 0)
        
        for item in self.all_predictions:
            subdf = pd.DataFrame(self.all_predictions[item], columns=['movieId', 'pred_rating'])
            subdf['userId'] = item
            cols = subdf.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            subdf = subdf[cols]        
            self.all_recommenddf = pd.concat([self.all_recommenddf, subdf], axis = 0)
        return rmse

    def predict(self, userId, movieId):
        uuid, iid, true_r, predict_r, details  = self.model.predict(userId, movieId)
        return uuid, iid, true_r, predict_r, details
    
    def cross_validate(self):
        print('Cross Validating the data...')
        cv_result = cross_validate(self.model, self.data, n_jobs=-1)
        cv_result = round(cv_result['test_rmse'].mean(),3)
        self.mean_cv_rmse = cv_result
        print('Mean CV RMSE is ' + str(cv_result))
        return cv_result
    
    def compute_similarities(self):
        return self.model.compute_similarities()

    def recommend(self, user_id):
        #print('Recommending top ' + str(self.n)+ ' products for userid : ' + str(user_id) + ' ...')
        df = self.recommenddf[self.recommenddf['userId'] == user_id]
        #display(df)
        return df
    
    def recommend_all(self, user_id):
        #print('All ratings for userid : ' + str(user_id) + ' ...')
        df = self.all_recommenddf[self.all_recommenddf['userId'] == user_id]
        #display(df)
        return df

