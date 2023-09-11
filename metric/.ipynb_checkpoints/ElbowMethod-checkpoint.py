import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

class elbowMethod():
    '''
    Train a Kmeans model with a list of n_clusters and calculate the distortion of each n_cluster 
    and show the corresponding elbow graph, and the distortion difference graph
    
     Methods
    -------
    run (Start=1, Stop=10, Step=1, max_iterations=300
        Trains Kmeans model with a list of n_clusters.
        the list of n_clusters is defined by start, stop and the steps, just like numpy.arrange
    showPlot (n=1.0)
        showing two plots
        1. A elbow method graph
        2. The differences in the distortion between two consecutive clusters
        We can set a maximum number of clusters by 'upto_cluster' argument.

    '''
    
    def __init__(self, matrix):
        self.matrix = matrix
        self.wcss = list()
        self.differences = list()
        self.clusters = list()

    def run(self, Start=2, Stop=10, Step=1, max_iterations=300):
        # defining a list of n_clusters with a start, stop and the steps, just like numpy.arange
        self.clusters = np.arange(Start, Stop, Step)
        # Initiating and training a Kmeans model for each value of n_cluster in the list above
        for i in self.clusters:
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=max_iterations, n_init='auto', random_state=0)
            kmeans.fit(self.matrix)
            # appending the distortions of each iteration in a list
            self.wcss.append(kmeans.inertia_ / np.size(self.matrix, axis=0))
        self.differences = list()
        # saving the difference between two consecutive clusters in the list "differences"
        for i in range(len(self.wcss) - 1):
            self.differences.append(self.wcss[i] - self.wcss[i + 1])

    def showPlot(self, boundary=5, upto_cluster=None):
        # showing the plots, we can set a top cluster by 'upto_cluster' argument
        if upto_cluster is None:
            WCSS = self.wcss
            DIFF = self.differences
        else:
            WCSS = self.wcss[:upto_cluster]
            DIFF = self.differences[:upto_cluster - 1]
        plt.figure(figsize=(15, 6))
        plt.subplot(131).set_title('Elbow Method Graph')
        plt.plot(self.clusters, WCSS)
        plt.grid(visible=True)
        plt.xticks(self.clusters)
        plt.subplot(132).set_title('Differences in Each Two Consective Clusters')
        len_differences = len(DIFF)
        X_differences = range(1, len_differences + 1)
        plt.plot(self.clusters[1:], DIFF)
        plt.plot(self.clusters[1:], np.ones(len_differences) * boundary, 'r')
        plt.plot(self.clusters[1:], np.ones(len_differences) * (-boundary), 'r')
        plt.xticks(self.clusters)
        plt.grid(visible=True)
        plt.show()
