import mock
import unittest

from porter.datascience import (BaseFeatureEngineer, BaseModel,
                             WrappedFeatureEngineer, WrappedModel)
from tests.utils import (KERAS_MODEL_PATH, KERAS_VALIDATE_PATH,
                         SKLEARN_MODEL_PATH, SKLEARN_VALIDATE_PATH)


class TestWrappedModel(unittest.TestCase):
    def test_predict(self):
        class MockModel:
            def predict(self, X):
                return X + 1
        model = WrappedModel(MockModel(), 'test-model-name', 'test-model-id')
        actual = model.predict(1)
        expected = 2
        self.assertEqual(actual, expected)

    def test_from_file_sklearn(self):
        pass

    def test_from_file_keras(self):
        pass


class TestWrappedFeatureEngineer(unittest.TestCase):
    def test_transform(self):
        class MockTransformer:
            def transform(self, X):
                return X + 1
        feature_engineer = WrappedFeatureEngineer(MockTransformer())
        actual = feature_engineer.transform(1)
        expected = 2
        self.assertEqual(actual, expected)


class TestLoadFunctions(unittest.TestCase):
    def test_load_pkl(self):
        pass

    def test_load_h5(self):
        pass

    def test_load_file(self):
        pass


if __name__ == '__main__':
    unittest.main()
