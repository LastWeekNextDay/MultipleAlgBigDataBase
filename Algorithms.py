from KNN import KNN
from Visualization import Visualization
from MeanShift import MeanShift


class Algorithms:
    @staticmethod
    def knn_rand(points_list):
        # ---------------RANDOM------------------#
        # Using the 'random' algorithm
        knn = KNN(alg='random', cluster_num=3)
        knn.fit(points_list)
        clustered_data = knn.cluster()

        # Visualization of KNN random
        centroids_list = knn.get_all_centroids()
        iteration = 1
        print("Centroid iterations in KNN Random Algorithm")
        for each in centroids_list:
            print("Iteration " + str(iteration) + ":")
            print(str(each))
            iteration += 1
        visualization = Visualization(title='KNN Random', x_label='Peak Players', y_label='Average players')
        visualization.init(clustered_data[:, 0], clustered_data[:, 1], clustered_data[:, 2], knn.centroids)
        visualization.show()

    @staticmethod
    def knn_kmeanspp(points_list):
        # ---------------K-Means++------------------#
        # Using the 'K-Means++' algorithm
        knn = KNN(alg='kmeans++', cluster_num=3)
        knn.fit(points_list)
        clustered_data = knn.cluster()
        # Visualization of K-Means++
        visualization = Visualization(title='KNN K-Means++', x_label='Peak Players', y_label='Average players')
        visualization.init(clustered_data[:, 0], clustered_data[:, 1], clustered_data[:, 2])
        visualization.show()

    @staticmethod
    def meanshift(points_list):
        # ---------------MeanShift------------------#
        # MeanShift clustering
        meanshift = MeanShift(bandwidth=2)
        clustered_data = meanshift.cluster(points_list)
        # Visualization of MeanShift
        visualization = Visualization(title='MeanShift', x_label='Peak Players', y_label='Average players')
        visualization.init(clustered_data[:, 0], clustered_data[:, 1], clustered_data[:, 2])
        visualization.show()

