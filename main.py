import numpy as np
import pandas as pd
from Algorithms import Algorithms

if __name__ == '__main__':
    # Load the dataset
    file_path = r'Valve_Player_Data.csv'
    data = pd.read_csv(file_path)

    # 'Avg_players' and 'Peak_Players'
    X_set = data['Avg_players'].values
    Y_set = data['Peak_Players'].values
    points_list = [[x, y] for x, y in zip(X_set, Y_set)]

    Algorithms.knn_rand(points_list)
    Algorithms.knn_kmeanspp(points_list)
    Algorithms.dbscan(points_list)

