import numpy as np


class DBSCAN:
    EPS = 0.5

    def __init__(self, min_samples=5):
        self.min_samples = min_samples
        self.labels = []

    def fit_predict(self, points_list):
        points = np.array(points_list)
        self.labels = [-1] * len(points)
        cluster_id = 0

        for i in range(len(points)):
            if self.labels[i] != -1:
                continue  # already labeled

            neighbors = self._region_query(points, i)

            if len(neighbors) < self.min_samples:
                self.labels[i] = -1  # mark as noise
            else:
                self._expand_cluster(points, neighbors, cluster_id)
                cluster_id += 1

        return self.labels

    def _region_query(self, points, point_idx):
        neighbors = []
        for i in range(len(points)):
            if np.linalg.norm(points[point_idx] - points[i]) < self.EPS:
                neighbors.append(i)
        return neighbors

    def _expand_cluster(self, points, neighbors, cluster_id):
        queue = neighbors[:]
        self.labels[queue[0]] = cluster_id

        for idx in queue:
            if self.labels[idx] == -1:  # previously marked as noise
                self.labels[idx] = cluster_id

            if self.labels[idx] != -1:
                continue  # already labeled

            self.labels[idx] = cluster_id
            point_neighbors = self._region_query(points, idx)

            if len(point_neighbors) >= self.min_samples:
                queue += point_neighbors
