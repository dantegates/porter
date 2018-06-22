import json
import os
import tempfile
import unittest

import keras
import numpy as np
import pandas as pd
import sklearn.preprocessing
from porter.services import _ID_KEY, NumpyEncoder
from sklearn.externals import joblib

HERE = os.path.dirname(__file__)


class TestExample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X = pd.DataFrame(
            data=np.random.randint(0, 100, size=(10, 4)),
            columns=[_ID_KEY] + ['feature1', 'feature2', 'column3'])
        cls.y = np.random.randint(1, 10, size=10)
        cls.preprocessor = sklearn.preprocessing.StandardScaler().fit(cls.X.drop(_ID_KEY, axis=1))
        cls.model = keras.models.Sequential([
            keras.layers.Dense(20, input_shape=(3,)),
            keras.layers.Dense(1)
        ])
        cls.model.compile(loss='mean_squared_error', optimizer='sgd')
        cls.model.fit(cls.preprocessor.transform(cls.X.drop(_ID_KEY, axis=1)), cls.y, verbose=0)
        cls.predictions = cls.model.predict(cls.preprocessor.transform(cls.X.drop(_ID_KEY, axis=1))).reshape(-1)

    def test(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            joblib.dump(self.preprocessor, os.path.join(tmpdirname, 'preprocessor.pkl'))
            keras.models.save_model(self.model, os.path.join(tmpdirname, 'model.h5'))
            with open(os.path.join(HERE, '../examples/example.py')) as f:
                example = f.read()
            namespace = {'model_directory': tmpdirname}
            exec(example, namespace)
        test_client = namespace['model_app'].app.test_client()
        app_input = self.X.to_dict('records')
        response = test_client.post('/supa-dupa-model/prediction', data=json.dumps(app_input, cls=NumpyEncoder))
        actual_response_data = json.loads(response.data)
        expected_model_id = 'supa-dupa-model-v0'
        expected_predictions = {
            id_: pred for id_, pred in zip(self.X[_ID_KEY], self.predictions)
        }
        self.assertEqual(actual_response_data['model_id'], expected_model_id)
        for rec in actual_response_data['predictions']:
            actual_id, actual_pred = rec[_ID_KEY], rec['prediction']
            expected_pred = expected_predictions[actual_id]
            self.assertTrue(np.allclose(actual_pred, expected_pred))


if __name__ == '__main__':
    unittest.main()
