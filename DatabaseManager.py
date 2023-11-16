import pandas as pd
import sys
import os

class DatabaseManager:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)

    def get_column(self, col_name):
        return self.data[col_name].values


