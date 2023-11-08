import numpy as np


class DBSCAN:
    EPS = 0.01

    def __init__(self, min_samples=5):
        self.min_samples = min_samples
        self.labels = []

    def cluster(self, points_list):
        points = np.array(points_list)  # Convert the list of points to a NumPy array for computation
        self.labels = np.full(shape=points.shape[0], fill_value=-1, dtype=int)
        cluster_id = 0
        for point_idx in range(points.shape[0]):
            if self.labels[point_idx] != -1:
                continue  # This point is already labeled
            neighbors = self._region_query(points, point_idx)
            if len(neighbors) < self.min_samples:
                self.labels[point_idx] = -1  # Label as noise
            else:
                self._expand_cluster(points, neighbors, cluster_id)
                cluster_id += 1
        return np.column_stack((points, self.labels))

    def _region_query(self, points, point_idx):
        neighbors = []
        point = points[point_idx]
        for idx, candidate_point in enumerate(points):
            if np.linalg.norm(point - candidate_point) < self.EPS:
                neighbors.append(idx)
        return neighbors

    def _expand_cluster(self, points, neighbors, cluster_id):
        self.labels[neighbors[0]] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            if self.labels[neighbor_idx] == -1:
                self.labels[neighbor_idx] = cluster_id  # Change noise to border point
            elif self.labels[neighbor_idx] == -2:
                # This should never happen with the given initialization of labels to -1
                pass
            else:
                # This point is already processed
                i += 1
                continue

            point_neighbors = self._region_query(points, neighbor_idx)
            if len(point_neighbors) >= self.min_samples:
                neighbors += point_neighbors
            i += 1
        return self.labels
