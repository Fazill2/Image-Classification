from threading import Lock

import sklearn

from models.random_classifier import RandomClassifier


class ModelHandler:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelHandler, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.models = {
            'Random': RandomClassifier
        }
        self.model_list = [key for key in self.models.keys()]
        self.trained_models = {}
        self._initialized = True


    def train(self, model, x, y):
        model_obj = self.models[model].create_instance()
        model_obj.fit(x, y)
        self.trained_models[model] = model_obj

    def test(self, model, x, y) -> dict:
        model_obj = self.trained_models[model]
        predictions = model_obj.test(x)
        accuracy = sklearn.metrics.accuracy_score(y, predictions)
        f1 = sklearn.metrics.f1_score(y, predictions, average='macro')
        precision = sklearn.metrics.precision_score(y, predictions, average='macro')
        recall = sklearn.metrics.recall_score(y, predictions, average='macro')
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
        }

    def predict(self, model, x):
        model_obj = self.trained_models[model]
        return model_obj.predict(x)