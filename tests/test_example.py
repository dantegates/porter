import json
import os
import tempfile
import unittest

import keras
import numpy as np
import sklearn.preprocessing
from sklearn.externals import joblib

HERE = os.path.dirname(__file__)


class TestExample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X = np.random.rand(10, 4)  # extra column for ID
        cls.y = np.random.randint(1, 10, size=10)
        cls.preprocessor = sklearn.preprocessing.StandardScaler().fit(cls.X[:,1:])
        cls.model = keras.models.Sequential([
            keras.layers.Dense(20, input_shape=(3,)),
            keras.layers.Dense(1)
        ])
        cls.model.compile(loss='mean_squared_error', optimizer='sgd')
        cls.model.fit(cls.preprocessor.transform(cls.X[:,1:]), cls.y, verbose=0)
        cls.predictions = cls.model.predict(cls.preprocessor.transform(cls.X[:,1:]))
        super().setUpClass()

    def test(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            joblib.dump(self.preprocessor, os.path.join(tmpdirname, 'preprocessor.pkl'))
            keras.models.save_model(self.model, os.path.join(tmpdirname, 'model.h5'))
            with open(os.path.join(HERE, '../example.py')) as f:
                example = ''.join(
                    L.format(model_directory=tmpdirname) if 'model_directory' in L else L
                    for L in f)
            namespace = {}
            exec(example, namespace)
        test_client = namespace['model_app'].app.test_client()
        cols = namespace['input_schema'] + ['id']
        app_input = [{c: v for c, v in zip(cols, rec)} for rec in self.X]
        response = test_client.post('/supa-dupa-model/prediction', data=json.dumps(app_input))
        print(response)


if __name__ == '__main__':
    unittest.main()
