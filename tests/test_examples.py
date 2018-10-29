import json
import os
import tempfile
import unittest
from unittest import mock

import keras
import numpy as np
import pandas as pd
import sklearn.preprocessing

from sklearn.externals import joblib

from porter.utils import NumpyEncoder


HERE = os.path.dirname(__file__)


def load_example(filename, init_namespace=None):
    if init_namespace is None:
        init_namespace = {}
    with open(filename) as f:
        example = f.read()
    exec(example, init_namespace)
    return init_namespace


@mock.patch('porter.services.BaseService._ids', set())
class TestExample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X = pd.DataFrame(
            data=np.random.randint(0, 100, size=(10, 4)),
            columns=['id', 'feature1', 'feature2', 'column3'])
        cls.y = np.random.randint(1, 10, size=10)   
        cls.preprocessor = sklearn.preprocessing.StandardScaler().fit(cls.X.drop('id', axis=1))
        cls.model = keras.models.Sequential([
            keras.layers.Dense(20, input_shape=(3,)),
            keras.layers.Dense(1)
        ])
        cls.model.compile(loss='mean_squared_error', optimizer='sgd')
        cls.model.fit(cls.preprocessor.transform(cls.X.drop('id', axis=1)), cls.y, verbose=0)
        cls.predictions = cls.model.predict(cls.preprocessor.transform(cls.X.drop('id', axis=1))).reshape(-1)

    def test(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            joblib.dump(self.preprocessor, os.path.join(tmpdirname, 'preprocessor.pkl'))
            keras.models.save_model(self.model, os.path.join(tmpdirname, 'model.h5'))
            init_namespace = {'model_directory': tmpdirname}
            namespace = load_example(os.path.join(HERE, '../examples/example.py'), init_namespace)
        test_client = namespace['model_app'].app.test_client()
        app_input = self.X.to_dict('records')
        response = test_client.post('/supa-dupa-model/prediction', data=json.dumps(app_input, cls=NumpyEncoder))
        actual_response_data = json.loads(response.data)
        expected_model_name = 'supa-dupa-model'
        expected_model_version = '1.0.0'
        expected_predictions = {
            id_: pred for id_, pred in zip(self.X['id'], self.predictions)
        }
        self.assertEqual(actual_response_data['model_name'], expected_model_name)
        self.assertEqual(actual_response_data['model_version'], expected_model_version)
        for rec in actual_response_data['predictions']:
            actual_id, actual_pred = rec['id'], rec['prediction']
            expected_pred = expected_predictions[actual_id]
            self.assertTrue(np.allclose(actual_pred, expected_pred))


@mock.patch('porter.services.BaseService._ids', set())
class TestExampleHealthCheckEndponts(unittest.TestCase):
    def test(self):
        # just testing that the example can be executed
        namespace = load_example(os.path.join(HERE, '../examples/health_check_endpoints.py'))


@mock.patch('porter.services.BaseService._ids', set())
class TestExampleHealthCheckEndponts(unittest.TestCase):
    def test(self):
        # just testing that the example can be executed
        namespace = load_example(os.path.join(HERE, '../examples/middleware.py'))


if __name__ == '__main__':
    unittest.main()
