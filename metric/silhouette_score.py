import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
class SilhouetteAnalyzer:
    def __init__(self, matrix):
        self.matrix = matrix
        self.s_scores = list()
        self.clusters = list()

    def run(self, start=1, stop=10, step=1, max_iterations=300):
        # defining a list of n_clusters with a start, stop and the steps, just like numpy.arrange
        self.clusters = np.arange(start, stop, step)
        # initiating and training a Kmeans model with for each value of n_cluster in the list above
        for i in self.clusters:
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=max_iterations, n_init='auto', random_state=0)
            kmeans.fit(self.matrix)
            self.s_scores.append(silhouette_score(self.matrix, kmeans.labels_, metric='sqeuclidean'))

    def showPlot(self, boundary=500, upto_cluster=None):
        if upto_cluster is None:
            S_SCORE = self.s_scores
        else:
            S_SCORE = self.s_scores[:upto_cluster]
        plt.plot(self.clusters, S_SCORE)
        plt.title('silhouette coef. as a function of the corresponding n_clusters')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette coefficient')
        plt.grid()
