import numpy as np
import pandas as pd
import os.path
from sklearn import preprocessing
from collections import defaultdict
from surprise import Dataset, Reader, SVD, SVDpp, KNNBasic, KNNWithMeans, dump, accuracy
from surprise.model_selection import cross_validate
from sklearn.preprocessing import MinMaxScaler

movie_df = pd.read_csv('./data/external/movies.csv')
orig_ratings = pd.read_csv('./data/external/ratings.csv')
avg_ratings = orig_ratings.groupby('movieId')['rating'].mean().reset_index()
amnt_ratings = orig_ratings.groupby('movieId')['rating'].count().reset_index()
movie_df = movie_df.merge( avg_ratings,on='movieId', how='outer')
movie_df.rename(columns = {'rating': 'avg_rating'}, inplace=True)
movie_df = movie_df.merge( amnt_ratings,on='movieId', how='outer')
movie_df.rename(columns = {'rating': 'amnt_rating'}, inplace=True)
movie_df['amnt_rating_inverse'] = 1/(2*movie_df['amnt_rating'])

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

def train_KNN_CFWeights(trainset, testset, data):
    print("Training KNN-based memory based model for hybrid...")
    sim_options = {'name': 'pearson',
                   'min_support': 30,
                   'user_based': True}

    knn = KNNBasic(k=30,sim_options=sim_options)
    knn.fit(trainset)
    knn_predictions = knn.test(testset)
    
    #cross-validation
    cv_result = cross_validate(knn, data, n_jobs=-1)
    cv_result = round(cv_result['test_rmse'].mean(),3)
    print('Mean CV RMSE is ' + str(cv_result))
    # save model and its predictions to the disk
    dump.dump('model/trained_models/KNN_CFWeights',algo=knn,predictions=knn_predictions)
    return knn_predictions, knn
    
    
def train_CF_Model(trainset, testset, data):
    print("Training and saving a CF model ...")
    algo = SVD(n_factors=170, n_epochs=90,lr_all=0.005, reg_all=0.05)
    algo.fit(trainset)
    algo_predictions = algo.test(testset)
    
    #cross-validation
    cv_result = cross_validate(algo, data, n_jobs=-1)
    cv_result = round(cv_result['test_rmse'].mean(),3)
    print('Mean CV RMSE is ' + str(cv_result))
    # save model and its predictions to the disk
    dump.dump('model/trained_models/CF_Model',algo=algo,predictions=algo_predictions)
    return algo_predictions, algo
    

#train and save the models
def train_cf_models():
    #prepare data
    trainset, testset, data = prepare_data()
    cf_predictions, cf_algo = train_CF_Model(trainset, testset, data)
    knn_predictions, knn = train_KNN_CFWeights(trainset, testset, data)
    print("Training is done!")
    return trainset, testset, data, cf_algo, cf_predictions, knn, knn_predictions    

def get_collaborative_filtering_weights(trainset = None, similarity_mat = None, threshold = 0, algo=None):
    if algo is None:
        knn_predictions, knn = dump.load('./model/trained_models/KNN_CFWeights')
        
    if trainset is None:
        trainset, ___, ___ = prepare_data()
        
    if similarity_mat is None:
        similarity_mat = knn.compute_similarities()
        
    a = np.empty(len(similarity_mat))
    for i in range(len(similarity_mat)):
        a[i] = (similarity_mat[i] > 0).sum() -1
    weights = a / a.max()
    return weights

def order_test_results(predictions, n = None):
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        if (n is None):
            top_n[uid] = user_ratings
        else:
            top_n[uid] = user_ratings[:n]
    return top_n

class CollaborativeFilteringRecommender():
    """
    Collaborative Filtering Class
    """
    def __init__(self, predictions=None, model=None):
        self.model = model
        self.trainset = None
        self.testset = None
        self.data = None
        self.pred_test = predictions
        self.recommenddf = None
        self.predictions = None
        self.mean_cv_rmse = None
        self.cv_result = None
        
        path = './data/processed/CFMatrix.csv'
        if os.path.isfile(path) == True:
            self.recommenddf = pd.read_csv(path)
            
    def recompute_surprise_data(self):
        #prepare data
        self.trainset, self.testset, self.data = prepare_data()

    def fit_and_predict(self):
        if (self.model is None or self.pred_test is None):
            #print('Fitting the train data...')
            self.model = SVD(n_factors=170, n_epochs=90,lr_all=0.005, reg_all=0.05)
            self.model.fit(self.trainset)       

            #print('Predicting the test data...')
            self.pred_test = self.model.test(self.testset)
            dump.dump('model/trained_models/CF_Model',algo= self.model ,predictions=self.pred_test)
            
        #rmse = round(accuracy.rmse(self.pred_test), 3)
        #print('RMSE for the predicted result is ' + str(rmse))
        
        self.predictions = order_test_results(self.pred_test)
        
        self.recommenddf = pd.DataFrame(columns=['userId', 'movieId', 'rating'])
        for item in self.predictions:
            subdf = pd.DataFrame(self.predictions[item], columns=['movieId', 'rating'])
            subdf['userId'] = item
            cols = subdf.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            subdf = subdf[cols]          
            self.recommenddf = pd.concat([self.recommenddf, subdf], axis = 0)
        
        #self.recommenddf = self.recommenddf[self.recommenddf.amnt_rating_inverse != 1]
        #self.recommenddf.sort_values(by=['cf_score'], ascending=[False])
        #self.recommenddf['cf_score'] = scaler.fit_transform(self.recommenddf.rating.values.reshape(-1, 1))
        
        # save to the harddrive:
        self.recommenddf.to_csv('./data/processed/CFMatrix.csv')
        print("Done calculating predictions and scores!")

    def predict(self, userId, movieId):
        uuid, iid, true_r, predict_r, details  = self.model.predict(userId, movieId)
        return uuid, iid, true_r, predict_r, details
    
    def cross_validate(self):
        print('Cross Validating the data...')
        cv_result = cross_validate(self.model, self.data, n_jobs=-1)
        self.cv_result = cv_result
        self.mean_cv_rmse = round(cv_result['test_rmse'].mean(),3)
        print('Mean CV RMSE is ' + str(self.mean_cv_rmse))
        print('Mean CV MAE is ' + str(round(cv_result['test_mae'].mean(),3)))
        return cv_result
    
    def recommend(self, user_id, n1=50, n2=20, version = 0):
        df = self.recommenddf[self.recommenddf['userId'] == user_id]
        df = df.merge(movie_df, on='movieId', how='inner')
        scaler = MinMaxScaler()
        if version == 1:
            df = df.head(n1)
            df.loc[:,['rating', 'avg_rating', 'amnt_rating', 'amnt_rating_inverse']] = scaler.fit_transform(df[['rating', 'avg_rating', 'amnt_rating', 'amnt_rating_inverse']])
            #df[['rating', 'avg_rating', 'amnt_rating', 'amnt_rating_inverse']]= scaler.fit_transform(df[['rating', 'avg_rating', 'amnt_rating', 'amnt_rating_inverse']])
            df['cf_score'] = df.rating + df.avg_rating + df.amnt_rating_inverse
            df.sort_values(by=['cf_score'], ascending=[False], inplace=True)
        else:
            df['cf_score'] = scaler.fit_transform(df.rating.values.reshape(-1, 1))
        return df.head(n2)
    
    def get_rankings_for_movies(self, user_id, movies):
        df = self.recommenddf[(self.recommenddf["movieId"].isin(movies)) & (self.recommenddf['userId'] == user_id) ]
        return df
