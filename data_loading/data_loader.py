from threading import Lock

import pandas as pd
import os

from sklearn.model_selection import train_test_split


class DataLoader:
    _instance = None
    _lock = Lock()

    def __new__(cls, path: str, test_size: float = 0.2):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DataLoader, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, path: str, test_size: float =0.2):
        if self._initialized:
            return
        self.path: str = path
        self.train_df: pd.DataFrame = pd.DataFrame()
        self.test_df: pd.DataFrame = pd.DataFrame()
        self.train_y: pd.Series = pd.Series()
        self.test_y: pd.Series = pd.Series()
        self.test_size: float = test_size
        self._initialized = True
        self.class_names: list[str] = []

    def load(self):
        data = []
        image_dirs = os.listdir(self.path)
        for image_dir in image_dirs:
            files = os.listdir(self.path + '/' +  image_dir)
            files_labels = [[self.path + '/' +  image_dir + '/' + file, image_dir] for file in files]
            data.extend(files_labels)
        full_df = pd.DataFrame.from_records(data, columns=['file', 'label'])
        self.train_df, self.test_df = train_test_split(full_df, test_size=self.test_size)
        self.train_y = self.train_df['label']
        self.test_y = self.test_df['label']

