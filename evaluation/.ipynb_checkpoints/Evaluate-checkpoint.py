import numpy as np
import pandas as pd
import random

class Evaluate():
    def __init__(self, model, trainset):
        self.model = model
        self.trainset = trainset

    def Variety(self, number_of_users = 100, number_of_recommendations = 10):
        variety = []
        unique_movies = set()
        for i in range(10):
            # Select users randomly
            selection_of_inner_users = np.array(random.sample(self.trainset.all_users(), number_of_users))
            selection_of_users = np.empty(number_of_users)
            for i in range(len(selection_of_inner_users)):
                u = self.trainset.to_raw_uid(selection_of_inner_users[i])
                selection_of_users[i] = u
                # While computing 10 recommendatiosn for each, count only unique movies
                recommendations = self.model.recommend(u, number_of_recommendations)
                movie_list = set(recommendations.movieId.values.tolist())
                unique_movies = unique_movies | movie_list
            
            variety.append(len(sorted(unique_movies)))
        return 0.001*np.array(variety).mean()-0.01, 0.001*np.array(variety).std()
    
    def Coverage(self):
        return len(self.model.pred_test) / len(self.testset)


