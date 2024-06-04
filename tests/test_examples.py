import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import sklearn.preprocessing
import keras
from porter.utils import NumpyEncoder
import joblib

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
        X = np.random.rand(10, 3)
        cls.X = pd.DataFrame(
            data=X,
            columns=['feature1', 'feature2', 'column3'])
        cls.X['id'] = range(len(X))
        cls.y = np.random.randint(1, 10, size=10)
        cls.preprocessor = sklearn.preprocessing.StandardScaler().fit(cls.X.drop('id', axis=1))
        cls.model = keras.models.Sequential([
            keras.layers.Dense(20, activation='relu', input_shape=(3,)),
            keras.layers.Dense(1, activation='relu')
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
        response = test_client.post('/supa-dupa-model/v1/prediction', data=json.dumps(app_input, cls=NumpyEncoder))
        actual_response_data = json.loads(response.data)
        expected_model_name = 'supa-dupa-model'
        expected_api_version = 'v1'
        expected_predictions = {
            id_: pred for id_, pred in zip(self.X['id'], self.predictions)
        }
        self.assertEqual(actual_response_data['model_context']['model_name'], expected_model_name)
        self.assertEqual(actual_response_data['model_context']['api_version'], expected_api_version)
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
class TestAPILogging(unittest.TestCase):
    def test(self):
        # just testing that the example can be executed
        namespace = load_example(os.path.join(HERE, '../examples/api_logging.py'))


@mock.patch('porter.services.BaseService._ids', set())
class TestGettingStarted(unittest.TestCase):
    def test(self):
        # just testing that the example can be executed
        namespace = load_example(os.path.join(HERE, '../examples/getting_started.py'))        


@mock.patch('porter.services.BaseService._ids', set())
class TestFunctionService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = load_example(os.path.join(HERE, '../examples/function_service.py'))
        cls.test_app = cls.ns['app'].app.test_client()

    def test_get(self):
        r = self.test_app.get('/math/v1/sum')
        self.assertEqual(r.status_code, 200)
        r = self.test_app.get('/math/v1/prod')
        self.assertEqual(r.status_code, 200)

    def test_post_sum(self):
        endpoint = '/math/v1/sum'
        a = list(range(1, 11))
        r = self.test_app.post(endpoint, data=json.dumps(a))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(sum(a), int(r.data))

    def test_post_prod(self):
        endpoint = '/math/v1/prod'
        a = list(range(1, 11))
        r = self.test_app.post(endpoint, data=json.dumps(a))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(np.prod(a), int(r.data))

        # test additional_checks: zero not allowed
        a = list(range(0, 11))
        r = self.test_app.post(endpoint, data=json.dumps(a))
        print(r.json)
        self.assertEqual(r.status_code, 422)


@mock.patch('porter.services.BaseService._ids', set())
class TestContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = load_example(os.path.join(HERE, '../examples/contracts.py'))
        cls.test_app = cls.ns['model_app'].app.test_client()

    def test_instance_prediction_service(self):
        """Configurations for this service are:

        endpoint = /datascience/user-ratings/v2/instancePrediction
        validate_request_data = True
        batch_prediction = False

        The description of the input schema is

        {
            "id": Integer(),
            "user_id": Integer(),
            "title_id": Integer(),
            "genre": String(['comedy', 'action', 'drama']),
            "average_rating": Number(min=0, max=10)
        }

        """
        valid_data = {
            "id": 0,
            "user_id": 1,
            "title_id": 19302943284,
            "genre": "action",
            "average_rating": 7.9
        }

        invalid_data_missing_key = {
            "user_id": 1,
            "title_id": 19302943284,
            "genre": "action",
            "average_rating": 7.9
        }

        invalid_data_invalid_genre = {
            "id": 0,
            "user_id": 1,
            "title_id": 19302943284,
            "genre": "isnotvalid",
            "average_rating": 7.9
        }

        invalid_data_invalid_average_rating = {
            "id": 0,
            "user_id": 1,
            "title_id": 19302943284,
            "genre": "action",
            "average_rating": -1
        }

        endpoint = '/datascience/user-ratings/v2/instancePrediction'
        r = self.test_app.post(endpoint, data=json.dumps(valid_data))
        self.assertEqual(r.status_code, 200)

        r = self.test_app.post(endpoint, data=json.dumps(invalid_data_missing_key))
        self.assertEqual(r.status_code, 422)
        self.assertIn("data must contain ['id']", r.json['error']['messages'][0])

        r = self.test_app.post(endpoint, data=json.dumps(invalid_data_invalid_genre))
        self.assertEqual(r.status_code, 422)
        self.assertIn('genre', r.json['error']['messages'][0])

        r = self.test_app.post(endpoint, data=json.dumps(invalid_data_invalid_average_rating))
        self.assertEqual(r.status_code, 422)
        self.assertIn('average_rating', r.json['error']['messages'][0])

    def test_batch_prediction_service(self):
        """Configurations for this service are:

        endpoint = /datascience/user-ratings/v2/prediction
        validate_request_data = True
        batch_prediction = True

        The description of the input schema is

        [
            {
                "id": Integer(),
                "user_id": Integer(),
                "title_id": Integer(),
                "genre": String(['comedy', 'action', 'drama']),
                "average_rating": Number(min=0, max=10)
            },
            ...
        ]

        """
        valid_data = [
            {
                "id": 0,
                "user_id": 1,
                "title_id": 19302943284,
                "genre": "action",
                "average_rating": 7.9
            },
            {
                "id": 0,
                "user_id": 1,
                "title_id": 19302943284,
                "genre": "action",
                "average_rating": 7.9
            },
            {
                "id": 0,
                "user_id": 1,
                "title_id": 19302943284,
                "genre": "action",
                "average_rating": 7.9
            },
        ]

        invalid_data_title_id_is_str = [
            {
                "id": 0,
                "user_id": 1,
                "title_id": 19302943284,
                "genre": "action",
                "average_rating": 7.9
            },
            {
                "id": 0,
                "user_id": 1,
                "title_id": 'a',
                "genre": "action",
                "average_rating": 7.9
            },
            {
                "id": 0,
                "user_id": 1,
                "title_id": 19302943284,
                "genre": "action",
                "average_rating": 7.9
            },
        ]

        invalid_data_not_an_array = {
            "id": 0,
            "user_id": 1,
            "title_id": 19302943284,
            "genre": "action",
            "average_rating": 7.9
        }

        endpoint = '/datascience/user-ratings/v2/prediction'
        r = self.test_app.post(endpoint, data=json.dumps(valid_data))
        self.assertEqual(r.status_code, 200)

        r = self.test_app.post(endpoint, data=json.dumps(invalid_data_title_id_is_str))
        self.assertEqual(r.status_code, 422)
        self.assertIn('data[1].title_id', r.json['error']['messages'][0])

        r = self.test_app.post(endpoint, data=json.dumps(invalid_data_not_an_array))
        self.assertEqual(r.status_code, 422)        
        self.assertIn('data must be array', r.json['error']['messages'][0])

    def test_probabilistic_service(self):
        """Configurations for this service are:

        endpoint = /datascience/proba-model/v3/prediction
        validate_request_data = False
        batch_prediction = True

        The description of the input schema is

        [
            {
                "id": Integer(),
                "user_id": Integer(),
                "title_id": Integer(),
                "genre": String(['comedy', 'action', 'drama']),
                "average_rating": Number(min=0, max=10)
            },
            ...
        ]
        """

        # this endpoint doesn't do data validations so we should get a 500 by sending
        # bad data instead of a 422
        invalid_data = {
            'not the data': 'you were looking for'
        }
        r = self.test_app.post('/datascience/proba-model/v3/prediction', data=json.dumps(invalid_data))
        self.assertEqual(r.status_code, 500)

    def test_spark_interface_service(self):
        """Configurations for this service are:

        endpoint = /datascience/batch-ratings-model/v1/prediction
        validate_request_data = False
        batch_prediction = True

        The description of the input schema is

        [
            {
                "id": Integer(),
                "user_id": Integer(),
                "title_id": Integer(),
                "genre": String(['comedy', 'action', 'drama']),
                "average_rating": Number(min=0, max=10)
            },
            ...
        ]
        """
        valid_data = [
            {
                "id": 0,
                "user_id": 1,
                "title_id": 19302943284,
                "genre": "action",
                "average_rating": 7.9
            },
            {
                "id": 0,
                "user_id": 1,
                "title_id": 19302943284,
                "genre": "action",
                "average_rating": 7.9
            },
            {
                "id": 0,
                "user_id": 1,
                "title_id": 19302943284,
                "genre": "action",
                "average_rating": 7.9
            },
        ]
        r = self.test_app.post('/datascience/batch-ratings-model/v1/prediction', data=json.dumps(valid_data))
        self.assertEqual(r.status_code, 202)
        self.ns['spark_interface_response_schema'].validate(json.loads(r.data))

    def test_custom_service_validations_fail(self):
        """Configurations for this service are:

        endpoint = /custom-service/v1/foo
        validate_request_data = True
        batch_prediction = N/A
        """
        invalid_data1 = 'this string is definitely not an object'
        invalid_data2 = {'this object': 'is an object', 'but not a': 'correct schema'}
        invalid_data_checking_nested_validations1 = {
            'string_with_enum_prop': 'a',
            'an_array': [1, 2],
            'another_property': {'a': 'a', 'b': 1},
            'yet_another_property': [
                {'foo': 'a'},
                {'foo': 'a', 'bar': 1}  # all props should be str
            ]
        }
        invalid_data_checking_nested_validations2 = {
            'string_with_enum_prop': 'a',
            'an_array': [1, 2],
            'another_property': {'a': 'a', 'b': 'not an int'},
            'yet_another_property': [
                {'foo': 'a'},
                {'foo': 'a', 'bar': '1'}
            ]
        }

        r = self.test_app.post('/custom-service/v1/foo', data=json.dumps(invalid_data1))
        self.assertEqual(r.status_code, 422)
        self.assertIn('data must be object', r.json['error']['messages'][0])

        r = self.test_app.post('/custom-service/v1/foo', data=json.dumps(invalid_data2))
        self.assertEqual(r.status_code, 422)
        self.assertIn("data must contain ['an_array', 'another_property', 'string_with_enum_prop', 'yet_another_property']", r.json['error']['messages'][0])

        r = self.test_app.post('/custom-service/v1/foo', data=json.dumps(invalid_data_checking_nested_validations1))
        self.assertEqual(r.status_code, 422)
        self.assertIn("yet_another_property[1].bar", r.json['error']['messages'][0])

        r = self.test_app.post('/custom-service/v1/foo', data=json.dumps(invalid_data_checking_nested_validations2))
        self.assertEqual(r.status_code, 422)
        self.assertIn("another_property.b", r.json['error']['messages'][0])

    def test_custom_service_validations_succeed(self):
        """Configurations for this service are:

        endpoint = /custom-service/v1/foo
        validate_request_data = True
        batch_prediction = N/A
        """
        valid_data = {
            'string_with_enum_prop': 'a',
            'an_array': [1, 2],
            'another_property': {'a': 'a', 'b': 1},
            'yet_another_property': [
                {'foo': 'a'},
                {'foo': 'a', 'bar': 'b'}
            ]
        }
        r = self.test_app.post('/custom-service/v1/foo', data=json.dumps(valid_data))
        self.assertEqual(r.status_code, 200)



if __name__ == '__main__':
    unittest.main()
