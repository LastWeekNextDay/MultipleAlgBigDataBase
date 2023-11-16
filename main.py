from Algorithms import Algorithms
from DatabaseManager import DatabaseManager
import sys
import os

if __name__ == '__main__':
    # Load the dataset
    # databaseManager = DatabaseManager(r'Valve_Player_Data.csv')
    def get_base_dir():
        if getattr(sys, 'frozen', False):
            # If the application is run as a bundle, the PyInstaller bootloader
            # extends the sys module by a flag frozen=True and sets the app
            # path into variable _MEIPASS'.
            return sys._MEIPASS
        else:
            # Otherwise, we are running in a normal Python environment
            return os.path.dirname(os.path.abspath(__file__))
    # Use the function to get the correct base directory
    base_dir = get_base_dir()

    # Construct the full path to the CSV file
    csv_file_path = os.path.join(base_dir, 'Valve_Player_Data.csv')

    # Now, you initialize your DatabaseManager with csv_file_path
    databaseManager = DatabaseManager(csv_file_path)

    # 'Avg_players' and 'Peak_Players'
    X_set = databaseManager.get_column('Peak_Players')
    Y_set = databaseManager.get_column('Avg_players')
    points_list = [[x, y] for x, y in zip(X_set, Y_set)]

    Algorithms.knn_rand(points_list)
    Algorithms.knn_kmeanspp(points_list)
    Algorithms.meanshift(points_list)

