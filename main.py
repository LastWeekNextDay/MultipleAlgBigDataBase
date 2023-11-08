import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from KNN import KNN
from Visualization import Visualization
from DBSCAN import DBSCAN

if __name__ == '__main__':
    # Load the dataset
    file_path = r'C:\Users\LastWeek\PycharmProjects\MultipleAlgBigDataBase\Valve_Player_Data.csv'
    data = pd.read_csv(file_path)

    # 'Avg_players' and 'Peak_Players'
    X_set = data['Avg_players'].values
    Y_set = data['Peak_Players'].values
    points_list = [[x, y] for x, y in zip(X_set, Y_set)]

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
    Visualization1 = Visualization(title='KNN Random', x_label='Peak Players', y_label='Average players')
    Visualization1.init(clustered_data[:, 0], clustered_data[:, 1], clustered_data[:, 2], knn.centroids)
    Visualization1.show()

    # Using the 'K-Means++' algorithm
    knn = KNN(alg='kmeans++', cluster_num=3)
    knn.fit(points_list)
    clustered_data = knn.cluster()

    # Visualization of KNN K-Means++
    Visualization2 = Visualization(title='KNN K-Means++', x_label='Peak Players', y_label='Average players')
    Visualization2.init(clustered_data[:, 0], clustered_data[:, 1], clustered_data[:, 2], knn.centroids)
    Visualization2.show()

    # DBSCAN clustering
    dbscan = DBSCAN(min_samples=3)
    labels = dbscan.fit_predict(points_list)

    # Visualization of DBSCAN
    points = np.array(points_list)
    Visualization3 = Visualization(title='DBSCAN', x_label='Peak Players', y_label='Average players')
    Visualization3.init(points[:, 0], points[:, 1], labels)
    Visualization3.show()
