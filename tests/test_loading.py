import os
import tempfile
import unittest

import keras
import numpy as np
import sklearn.linear_model
from porter import loading
import joblib


class TestLoadingSklearn(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X = np.random.rand(10, 20)
        cls.y = np.sum(cls.X, axis=1) + np.random.randint(1, 10, size=10)
        cls.model = sklearn.linear_model.LinearRegression()
        cls.model.fit(cls.X, cls.y)
        cls.predictions = cls.model.predict(cls.X)
        super().setUpClass()

    def test_load_pkl(self):
        with tempfile.NamedTemporaryFile(suffix='.pkl') as tmp:
            joblib.dump(self.model, tmp.name)
            loaded_model = loading.load_pkl(tmp.name)
        actual_predictions = loaded_model.predict(self.X)
        expected_predictions = self.predictions
        self.assertTrue(np.allclose(actual_predictions, expected_predictions))

    def test_load_file_pkl(self):
        with tempfile.NamedTemporaryFile(suffix='.pkl') as tmp:
            joblib.dump(self.model, tmp.name)
            loaded_model = loading.load_file(tmp.name)
        actual_predictions = loaded_model.predict(self.X)
        expected_predictions = self.predictions
        self.assertTrue(np.allclose(actual_predictions, expected_predictions))


class TestLoadingKeras(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X = np.random.rand(10, 20)
        cls.y = np.random.randint(1, 10, size=10)
        cls.model = keras.models.Sequential([
            keras.layers.Dense(20, input_shape=(20,)),
            keras.layers.Dense(1)
        ])
        cls.model.compile(loss='mean_squared_error', optimizer='sgd')
        cls.model.fit(cls.X, cls.y, verbose=0)
        cls.predictions = cls.model.predict(cls.X)
        super().setUpClass()

    def test_load_keras(self):
        with tempfile.NamedTemporaryFile(suffix='.keras') as tmp:
            keras.models.save_model(self.model, tmp.name)
            loaded_model = loading.load_keras(tmp.name)
        actual_predictions = loaded_model.predict(self.X)
        expected_predictions = self.predictions
        self.assertTrue(np.allclose(actual_predictions, expected_predictions))

    def test_load_file_keras(self):
        with tempfile.NamedTemporaryFile(suffix='.keras') as tmp:
            keras.models.save_model(self.model, tmp.name)
            loaded_model = loading.load_file(tmp.name)
        actual_predictions = loaded_model.predict(self.X)
        expected_predictions = self.predictions
        self.assertTrue(np.allclose(actual_predictions, expected_predictions))

    def test_load_h5(self):
        with tempfile.NamedTemporaryFile(suffix='.h5') as tmp:
            keras.models.save_model(self.model, tmp.name)
            loaded_model = loading.load_keras(tmp.name)
        actual_predictions = loaded_model.predict(self.X)
        expected_predictions = self.predictions
        self.assertTrue(np.allclose(actual_predictions, expected_predictions))

    def test_load_file_h5(self):
        with tempfile.NamedTemporaryFile(suffix='.h5') as tmp:
            keras.models.save_model(self.model, tmp.name)
            loaded_model = loading.load_file(tmp.name)
        actual_predictions = loaded_model.predict(self.X)
        expected_predictions = self.predictions
        self.assertTrue(np.allclose(actual_predictions, expected_predictions))


if __name__ == '__main__':
    unittest.main()
