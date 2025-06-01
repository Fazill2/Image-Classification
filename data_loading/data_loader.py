from threading import Lock

import pandas as pd
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

IMG_SIZE = (224, 224)

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
        tqdm.pandas()
        full_df['image_array'] = full_df['file'].progress_apply(DataLoader.load_image)
        full_df = full_df[full_df['image_array'].notnull()]
        full_df = full_df.drop(columns=['file'], inplace=False)
        self.train_df, self.test_df = train_test_split(full_df, test_size=self.test_size)
        self.train_y = self.train_df['label']
        self.test_y = self.test_df['label']
        self.train_df = self.train_df['image_array']
        self.test_df = self.test_df['image_array']

    @staticmethod
    def load_requested_image(file):
        image_df = pd.DataFrame({'image_array': [DataLoader.load_image(file)]})
        return image_df['image_array']

    @staticmethod
    def load_image(path):
        try:
            img = Image.open(path).convert('RGB')
            img = img.resize(IMG_SIZE)
            return np.array(img)
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            return None