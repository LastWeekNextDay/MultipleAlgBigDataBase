import matplotlib.pyplot as plt


class Visualization:
    def __init__(self, title, x_label, y_label):
        self.centroids = None
        self.labels = None
        self.Y_set = None
        self.X_set = None
        self.title = title
        self.x_label = x_label
        self.y_label = y_label

    def init(self, x_set, y_set, labels, centroids=None):
        self.X_set = x_set
        self.Y_set = y_set
        self.labels = labels
        self.centroids = centroids

    def show(self):
        plt.figure(figsize=(10, 8))
        plt.title(self.title)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.scatter(self.X_set, self.Y_set, c=self.labels, cmap='viridis', marker='o')
        if self.centroids is not None:
            plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', marker='x', label='Centroids')
        plt.show()
