import numpy as np
from sklearn.neighbors import NearestNeighbors


class MeanShift:
    MAX_ITER = 100

    def __init__(self, bandwidth=2):
        self.bandwidth = bandwidth
        self.cluster_centers_ = []
        self.labels_ = None
        self.points = None

    def fit(self, points_list):
        self.points = np.array(points_list)
        shift_points = np.copy(self.points)

        for _ in range(self.MAX_ITER):
            # Shift each point to the mean of points within its neighborhood
            for i, point in enumerate(shift_points):
                # Compute distances from the current point to all other points
                distances = np.linalg.norm(self.points - point, axis=1)
                # Identify points within the bandwidth
                within_bandwidth = self.points[distances < self.bandwidth]
                # Compute the mean of points within the bandwidth
                shift_points[i] = np.mean(within_bandwidth, axis=0)

            # Check for convergence (if points have stopped moving)
            if np.max(np.linalg.norm(shift_points - self.points, axis=1)) < self.bandwidth / 100:
                break

            self.points = shift_points

        # Identify unique cluster centers
        self.cluster_centers_, unique_indices = np.unique(shift_points, axis=0, return_inverse=True)
        self.labels_ = unique_indices

    def cluster(self, points_list):
        self.fit(points_list)
        return np.column_stack((self.points, self.labels_))
