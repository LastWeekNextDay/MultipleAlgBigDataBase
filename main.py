from Algorithms import Algorithms
from DatabaseManager import DatabaseManager

if __name__ == '__main__':
    # Load the dataset
    databaseManager = DatabaseManager(r'Valve_Player_Data.csv')

    # 'Avg_players' and 'Peak_Players'
    X_set = databaseManager.get_column('Peak_Players')
    Y_set = databaseManager.get_column('Avg_players')
    points_list = [[x, y] for x, y in zip(X_set, Y_set)]

    Algorithms.knn_rand(points_list)
    Algorithms.knn_kmeanspp(points_list)
    Algorithms.meanshift(points_list)

