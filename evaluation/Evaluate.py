import numpy as np
import pandas as pd
import random

class Evaluate():
    def __init__(self, model):
        self.model = model
        self.users = self.model.all_recommenddf.groupby(by='userId').count().index.values.tolist()

    def Variety(self, number_of_users = 100, number_of_recommendations = 10):
        variety = []
        
        for i in range(100):
            # Select users randomly
            selection_of_users = random.sample(self.users, number_of_users)
            # While computing 10 recommendatiosn for each, count only unique movies
            unique_movies = set()
            for u in selection_of_users:
                recommendations = self.model.recommend(u).head(number_of_recommendations)
                movie_list = set(recommendations.movieId.values.tolist())
                unique_movies = unique_movies | movie_list
            variety.append(len(sorted(unique_movies)))
        
        return 0.001*np.array(variety).mean()-0.01, 0.001*np.array(variety).std()
    
    def Coverage(self):
        return len(self.model.pred_test) / len(self.testset)


