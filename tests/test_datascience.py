import mock
import unittest

from ipa.datascience import BaseModel, BaseFeatureEngineer
from tests.utils import SKLEARN_MODEL_PATH, KERAS_MODEL_PATH, \
                       SKLEARN_VALIDATE_PATH, KERAS_VALIDATE_PATH


class TestBaseModel(unittest. TestCase):
    def test_predict(self):
        class MockModel:
            def predict(self, X):
                return X + 1
        model = BaseModel(MockModel(), 'test-model-name', 'test-model-id')
        actual = model.predict(1)
        expected = 2
        self.assertEqual(actual, expected)

    def test_from_file_sklearn(self):
        pass

    def test_from_file_keras(self):
        pass


class TestBaseFeatureEngineer(unittest.TestCase):
    def test_transform(self):
        class MockTransformer:
            def transform(self, X):
                return X + 1
        feature_engineer = BaseFeatureEngineer(MockTransformer())
        actual = feature_engineer.transform(1)
        expected = 2
        self.assertEqual(actual, expected)


class TestFunctions(unittest.TestCase):
    def test_load_pkl(self):
        pass

    def test_load_h5(self):
        pass

    def test_load_file(self):
        pass


if __name__ == '__main__':
    unittest.main()
