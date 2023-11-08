import numpy as np


class CentroidInitializer:
    def __init__(self, method='random'):
        self.method = method

    def initialize(self, points, k):
        if self.method == 'random':
            centroids = points[np.random.choice(len(points), k, replace=False)]
        elif self.method == 'kmeans++':
            centroids = [points[np.random.randint(len(points))]]
            while len(centroids) < k:
                distances = np.array([min([np.inner(c-x, c-x) for c in centroids]) for x in points])
                probabilities = distances / distances.sum()
                cumulative_probabilities = probabilities.cumsum()
                r = np.random.rand()
                for i, cp in enumerate(cumulative_probabilities):
                    if r < cp:
                        centroids.append(points[i])
                        break
        return np.array(centroids)
