import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder

class SimpleCNNClassifier:
    def __init__(self, seed: int = 42):
        self.seed = seed
        tf.random.set_seed(seed)
        self.model = None
        self.label_encoder = LabelEncoder()
        self.input_shape = None
        self.fitted = False

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        x_train_np = np.stack(x_train.values)
        y_train_np = y_train.values
        self.input_shape = x_train_np.shape[1:]
        y_encoded = self.label_encoder.fit_transform(y_train_np )
        num_classes = len(self.label_encoder.classes_)

        # Normalize image pixels to [0, 1]
        x_train_np = x_train_np.astype("float32") / 255.0

        # Build the CNN model
        self.model = models.Sequential([
            layers.Input(shape=self.input_shape),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])

        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

        self.model.fit(x_train_np, y_encoded, epochs=5, batch_size=32, verbose=0)
        self.fitted = True

    def test(self, x_test: pd.DataFrame) -> pd.Series:
        if not self.fitted:
            raise ValueError("Model must be fitted before calling test().")

        x_test_np = np.stack(x_test.values).astype("float32") / 255.0
        predictions = self.model.predict(x_test_np , verbose=0)
        predicted_indices = np.argmax(predictions, axis=1)
        predicted_labels = self.label_encoder.inverse_transform(predicted_indices)
        return pd.Series(predicted_labels)

    def predict(self, x_test: pd.DataFrame) -> str:
        if not self.fitted:
            raise ValueError("Model must be fitted before calling predict().")
        x_test_np = np.stack(x_test.values).astype("float32") / 255.0
        if len(x_test_np.shape) == 3:  # Single image
            x_test_np = np.expand_dims(x_test_np, axis=0)

        prediction = self.model.predict(x_test_np, verbose=0)
        predicted_index = np.argmax(prediction[0])
        return self.label_encoder.inverse_transform([predicted_index])[0]

    @staticmethod
    def get_name():
        return "SimpleCNNClassifier"

    @staticmethod
    def create_instance():
        return SimpleCNNClassifier()
