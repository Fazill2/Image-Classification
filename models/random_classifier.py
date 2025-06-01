import pandas as pd
import numpy as np

class RandomClassifier:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.labels = []

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.labels = y_train.unique().tolist()

    def test(self, x_test: pd.DataFrame) -> pd.Series:
        res_arr = np.random.choice(self.labels, size=len(x_test), replace=True)
        return pd.Series(res_arr, index=x_test.index)

    def predict(self, x_test: pd.DataFrame) -> str:
        return np.random.choice(self.labels)

    @staticmethod
    def get_name():
        return "RandomClassifier"

    @staticmethod
    def create_instance():
        return RandomClassifier()


        