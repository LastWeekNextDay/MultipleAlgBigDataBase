from CentroidInitializer import CentroidInitializer
import numpy as np


class KNN:
    MAX_ITER = 100
    CONV_DIST = 0.001

    def __init__(self, alg='random', cluster_num=3):
        self.points = None
        self.cluster_num = cluster_num
        self.centroids = []
        self.all_centroids = []  # Store centroids from all iterations
        self.labels = []
        self.centroid_initializer = CentroidInitializer(method=alg)

    def fit(self, points_list):
        points = np.array(points_list)  # Convert list to NumPy array for computation
        self.points = points
        self.centroids = self.centroid_initializer.initialize(points, self.cluster_num)
        self.all_centroids.append(self.centroids)  # Store initial centroids

    def cluster(self):
        for _ in range(self.MAX_ITER):
            distances = np.array([[np.inner(c - x, c - x) for c in self.centroids] for x in self.points])
            self.labels = np.argmin(distances, axis=1)
            new_centroids = np.array([self.points[self.labels == i].mean(axis=0) for i in range(self.cluster_num)])
            self.all_centroids.append(new_centroids)  # Store centroids of this iteration
            if np.all(np.linalg.norm(self.centroids - new_centroids, axis=1) < self.CONV_DIST):
                break
            self.centroids = new_centroids
        return np.column_stack((self.points, self.labels))

    def get_all_centroids(self):
        # Return the list of all centroids from each iteration
        return self.all_centroids
