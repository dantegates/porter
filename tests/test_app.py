"""
Tests for the `.app` attribute belonging to an instance of `porter.ModelService`.
"""


import json
import re
import unittest
from unittest import mock

import flask
from porter import exceptions as exc
from porter import __version__
from porter.datascience import BaseModel, BasePostProcessor, BasePreProcessor
from porter.services import ModelApp, PredictionServiceConfig


class TestAppPredictions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_app = ModelApp()
        cls.app = cls.model_app.app.test_client()
        # define objects for model 1
        class Preprocessor1(BasePreProcessor):
            def process(self, X):
                X['feature2'] = X.feature2.astype(str)
                return X
        class Model1(BaseModel):
            feature2_map = {str(x+1): x for x in range(5)}
            def predict(self, X):
                return X['feature1'] * X.feature2.map(self.feature2_map)
        class Postprocessor1(BasePostProcessor):
            def process(self, X_input, X_preprocessed, predictions):
                return predictions * -1
        input_features1 = ['feature1', 'feature2']

        # define objects for model 2
        class Preprocessor2(BasePreProcessor):
            def process(self, X):
                X['feature3'] = range(len(X))
                return X
        class Model2(BaseModel):
            def predict(self, X):
                return X['feature1'] + X['feature3']
        input_features2 = ['feature1']
        def user_check(X):
            if (X.feature1 == 0).any():
                raise exc.InvalidModelInput

        # define objects for model 3
        class Model3(BaseModel):
            def predict(self, X):
                return X['feature1'] * -1
        input_features3 = ['feature1']

        # define configs and add services to app
        service_config1 = PredictionServiceConfig(
            model=Model1(),
            name='a-model',
            version='0.0.0',
            preprocessor=Preprocessor1(),
            postprocessor=Postprocessor1(),
            input_features=input_features1,
            allow_nulls=False,
            batch_prediction=True
        )
        service_config2 = PredictionServiceConfig(
            model=Model2(),
            name='another-model',
            version='0.1.0',
            preprocessor=Preprocessor2(),
            postprocessor=None,
            input_features=input_features2,
            allow_nulls=False,
            batch_prediction=True,
            check_request=user_check
        )
        service_config3 = PredictionServiceConfig(
            model=Model3(),
            name='model-3',
            version='0.0.0-alpha',
            preprocessor=None,
            postprocessor=None,
            input_features=input_features3,
            allow_nulls=False,
            batch_prediction=False
        )
        cls.model_app.add_service(service_config1)
        cls.model_app.add_service(service_config2)
        cls.model_app.add_service(service_config3)

    def test_prediction_success(self):
        post_data1 = [
            {'id': 1, 'feature1': 2, 'feature2': 1},
            {'id': 2, 'feature1': 2, 'feature2': 2},
            {'id': 3, 'feature1': 2, 'feature2': 3},
            {'id': 4, 'feature1': 2, 'feature2': 4},
            {'id': 5, 'feature1': 2, 'feature2': 5},
        ]
        post_data2 = [
            {'id': 1, 'feature1': 10},
            {'id': 2, 'feature1': 10},
            {'id': 3, 'feature1':  1},
            {'id': 4, 'feature1':  3},
            {'id': 5, 'feature1':  3},
        ]
        post_data3 = {'id': 1, 'feature1': 5}
        actual1 = self.app.post('/a-model/prediction', data=json.dumps(post_data1))
        actual1 = json.loads(actual1.data)
        actual2 = self.app.post('/another-model/prediction', data=json.dumps(post_data2))
        actual2 = json.loads(actual2.data)
        actual3 = self.app.post('/model-3/prediction', data=json.dumps(post_data3))
        actual3 = json.loads(actual3.data)
        expected1 = {
            'model_name': 'a-model',
            'model_version': '0.0.0',
            'predictions': [
                {'id': 1, 'prediction': 0},
                {'id': 2, 'prediction': -2},
                {'id': 3, 'prediction': -4},
                {'id': 4, 'prediction': -6},
                {'id': 5, 'prediction': -8},
            ]
        }
        expected2 = {
            'model_name': 'another-model',
            'model_version': '0.1.0',
            'predictions': [
                {'id': 1, 'prediction': 10},
                {'id': 2, 'prediction': 11},
                {'id': 3, 'prediction': 3},
                {'id': 4, 'prediction': 6},
                {'id': 5, 'prediction': 7},
            ]
        }
        expected3 = {
            'model_name': 'model-3',
            'model_version': '0.0.0-alpha',
            'predictions': {'id': 1, 'prediction': -5}
        }
        self.assertEqual(actual1, expected1)
        self.assertEqual(actual2, expected2)
        self.assertEqual(actual3, expected3)

    def test_prediction_bad_requests(self):
        # should be array when sent to model1
        post_data1 = {'id': 1, 'feature1': 2, 'feature2': 1}
        # should be single object when sent to model3
        post_data2 = [{'id': 1, 'feature1': 2}, {'id': 2, 'feature1': 2}]
        # missing model2 features
        post_data3 = [{'id': 1, 'feature2': 1},
                      {'id': 2, 'feature2': 2},
                      {'id': 3, 'feature2': 3}]
        # contains nulls 
        post_data4 = {'id': 1, 'feature1': None}
        # contains nulls 
        post_data5 = [{'id': 1, 'feature1': 1, 'feature2': 1},
                      {'id': 1, 'feature1': 1, 'feature2': None}]
        # contains 0 values that don't pass user check
        post_data6 = [{'id': 1, 'feature1': 1, 'feature2': 1},
                      {'id': 1, 'feature1': 0, 'feature2': 1}]
        actuals = [
            self.app.post('/a-model/prediction', data=json.dumps(post_data1)),
            self.app.post('/model-3/prediction', data=json.dumps(post_data2)),
            self.app.post('/another-model/prediction', data=json.dumps(post_data3)),
            self.app.post('/model-3/prediction', data=json.dumps(post_data4)),
            self.app.post('/a-model/prediction', data=json.dumps(post_data5)),
            self.app.post('/another-model/prediction', data=json.dumps(post_data6)),
        ]
        # check status codes
        self.assertTrue(all(actual.status_code == 400 for actual in actuals))
        # check that all objects have error key
        self.assertTrue(all('error' in json.loads(actual.data) for actual in actuals))
        # check response values
        expected_error_values = [
            {'name': 'InvalidModelInput'},
            {'name': 'InvalidModelInput'},
            {'name': 'RequestMissingFields'},
            {'name': 'RequestContainsNulls'},
            {'name': 'RequestContainsNulls'},
            {'name': 'InvalidModelInput'},
        ]
        for actual, expectations in zip(actuals, expected_error_values):
            actual_error_obj = json.loads(actual.data)['error']
            for key, value in expectations.items():
                self.assertEqual(actual_error_obj[key], value)
        

class TestAppHealthChecks(unittest.TestCase):
    def setUp(self):
        self.model_app = ModelApp()
        self.app = self.model_app.app.test_client()

    def test_liveness_live(self):
        resp = self.app.get('/-/alive')
        self.assertEqual(resp.status_code, 200)

    def test_readiness_not_ready(self):
        resp_alive = self.app.get('/-/alive')
        resp_ready = self.app.get('/-/ready')
        expected_data = {
            'porter_version': __version__,
            'services': {}
        }
        self.assertEqual(resp_alive.status_code, 200)
        self.assertEqual(resp_ready.status_code, 503)
        self.assertEqual(json.loads(resp_alive.data), expected_data)
        self.assertEqual(json.loads(resp_ready.data), expected_data)

    @mock.patch('porter.services.PredictionServiceConfig.__init__')
    @mock.patch('porter.services.ModelApp.add_prediction_service')
    def test_readiness_ready_ready1(self, mock_add_prediction_service, mock_init):
        mock_init.return_value = None
        cf = PredictionServiceConfig()
        cf.name = 'model1'
        cf.version = '1.0.0'
        cf.id = 'model1'
        cf.endpoint = '/model1/prediction'
        cf.meta = {'foo': 1, 'bar': 2}
        self.model_app.add_service(cf)
        resp_alive = self.app.get('/-/alive')
        resp_ready = self.app.get('/-/ready')
        expected_data = {
            'porter_version': __version__,
            'services': {
                'model1': {
                    'status': 'READY',
                    'name': 'model1',
                    'version': '1.0.0',
                    'endpoint': '/model1/prediction',
                    'meta': {'foo': 1, 'bar': 2}
                }
            }
        }
        self.assertEqual(resp_alive.status_code, 200)
        self.assertEqual(resp_ready.status_code, 200)
        self.assertEqual(json.loads(resp_alive.data), expected_data)
        self.assertEqual(json.loads(resp_ready.data), expected_data)

    @mock.patch('porter.services.PredictionServiceConfig.__init__')
    @mock.patch('porter.services.ModelApp.add_prediction_service')
    def test_readiness_ready_ready2(self, mock_add_prediction_service, mock_init):
        mock_init.return_value = None
        cf1 = PredictionServiceConfig()
        cf1.name = 'model1'
        cf1.version = '1.0.0'
        cf1.id = 'model1:1.0.0'
        cf1.endpoint = '/model1/prediction'
        cf1.meta = {'foo': 1, 'bar': 2}
        cf2 = PredictionServiceConfig()
        cf2.name = 'model2'
        cf2.version = '0.0.0'
        cf2.id = 'model2:0.0.0'
        cf2.endpoint = '/model2/prediction'
        cf2.meta = {'foo': 1}
        self.model_app.add_services(cf1, cf2)
        resp_alive = self.app.get('/-/alive')
        resp_ready = self.app.get('/-/ready')
        expected_data = {
            'porter_version': __version__,
            'services': {
                'model1:1.0.0': {
                    'status': 'READY',
                    'name': 'model1',
                    'version': '1.0.0',
                    'endpoint': '/model1/prediction',
                    'meta': {'foo': 1, 'bar': 2},
                },
                'model2:0.0.0': {
                    'status': 'READY',
                    'name': 'model2',
                    'version': '0.0.0',
                    'endpoint': '/model2/prediction',
                    'meta': {'foo': 1},
                }
            }
        }
        self.assertEqual(resp_alive.status_code, 200)
        self.assertEqual(resp_ready.status_code, 200)
        self.assertEqual(json.loads(resp_alive.data), expected_data)
        self.assertEqual(json.loads(resp_ready.data), expected_data)

    def test_root(self):
        resp = self.app.get('/')
        self.assertEqual(resp.status_code, 200)


class TestAppErrorHandling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # DO NOT set app.testing = True here
        # doing so *disables* error handling in the application and instead
        # passes errors on to the test client (in our case, instances of
        # unittest.TestCase).
        # In this class we actually want to test the applications error handling
        # and thus do not set this attribute.
        # See, http://flask.pocoo.org/docs/0.12/api/#flask.Flask.test_client
        cls.model_app = ModelApp()
        flask_app = cls.model_app.app
        @flask_app.route('/test-error-handling/', methods=['POST'])
        def test_error():
            flask.request.get_json(force=True)
            raise Exception('exceptional testing of exceptions')
        cls.app_test_client = flask_app.test_client()
        cls.add_failing_model_service()

    def test_bad_request(self):
        # note data is unreadable JSON, thus a BadRequest
        resp = self.app_test_client.post('/test-error-handling/', data='bad data')
        actual = json.loads(resp.data)
        expected = {
            'error': {
                'name': 'BadRequest',
                'messages': ['The browser (or proxy) sent a request that this server could not understand.'],
                # user_data is None when not passed or unreadable
                'user_data': None,
                'traceback': re.compile('.*raise\sBadRequest.*')
            }
        }
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(actual['error']['name'], expected['error']['name'])
        self.assertEqual(actual['error']['messages'], expected['error']['messages'])
        self.assertEqual(actual['error']['user_data'], expected['error']['user_data'])
        self.assertTrue(expected['error']['traceback'].search(actual['error']['traceback']))

    def test_not_found(self):
        resp = self.app_test_client.get('/not-found/')
        actual = json.loads(resp.data)
        expected = {
            'error': {
                'name': 'NotFound',
                'messages': ['The requested URL was not found on the server.  '
                             'If you entered the URL manually please check your spelling and '
                             'try again.'],
                'user_data': None,
                'traceback': re.compile('.*raise\sNotFound.*')
            }
        }
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(actual['error']['name'], expected['error']['name'])
        self.assertEqual(actual['error']['messages'], expected['error']['messages'])
        self.assertEqual(actual['error']['user_data'], expected['error']['user_data'])
        self.assertTrue(expected['error']['traceback'].search(actual['error']['traceback']))

    def test_method_not_allowed(self):
        resp = self.app_test_client.get('/test-error-handling/')
        actual = json.loads(resp.data)
        expected = {
            'error': {
                'name': 'MethodNotAllowed',
                'messages': ['The method is not allowed for the requested URL.'],
                'user_data': None,
                'traceback': re.compile('.*raise\sMethodNotAllowed.*')
            }
        }
        self.assertEqual(resp.status_code, 405)
        self.assertEqual(actual['error']['name'], expected['error']['name'])
        self.assertEqual(actual['error']['messages'], expected['error']['messages'])
        self.assertEqual(actual['error']['user_data'], expected['error']['user_data'])
        self.assertTrue(expected['error']['traceback'].search(actual['error']['traceback']))

    def test_internal_server_error(self):
        user_data = {"valid": "json"}
        resp = self.app_test_client.post('/test-error-handling/', data=json.dumps(user_data))
        actual = json.loads(resp.data)
        expected = {
            'error': {
                'name': 'Exception',
                'messages': ['exceptional testing of exceptions'],
                'user_data': user_data,
                'traceback': re.compile('.*raise\sException')
            }
        }
        self.assertEqual(resp.status_code, 500)
        self.assertEqual(actual['error']['name'], expected['error']['name'])
        self.assertEqual(actual['error']['messages'], expected['error']['messages'])
        self.assertEqual(actual['error']['user_data'], expected['error']['user_data'])
        self.assertTrue(expected['error']['traceback'].search(actual['error']['traceback']))

    @mock.patch('porter.services.ServePrediction._predict')
    def test_prediction_fails(self, mock__predict):
        mock__predict.side_effect = Exception('testing a failing model')
        user_data = {'some test': 'data'}
        resp = self.app_test_client.post('/failing-model/prediction', data=json.dumps(user_data))
        actual = json.loads(resp.data)
        expected = {
            'model_name': 'failing-model',
            'model_version': 'B',
            '1': 'one',
            'two': 2,
            'error': {
                'name': 'PredictionError',
                'messages': ['an error occurred during prediction'],
                'user_data': user_data,
                'traceback': re.compile(".*testing\sa\sfailing\smodel.*"),
            }
        }
        self.assertEqual(resp.status_code, 500)
        self.assertEqual(actual['model_name'], expected['model_name'])
        self.assertEqual(actual['model_version'], expected['model_version'])
        self.assertEqual(actual['1'], expected['1'])
        self.assertEqual(actual['two'], expected['two'])
        self.assertEqual(actual['error']['name'], expected['error']['name'])
        self.assertEqual(actual['error']['messages'], expected['error']['messages'])
        self.assertEqual(actual['error']['user_data'], expected['error']['user_data'])
        self.assertTrue(expected['error']['traceback'].search(actual['error']['traceback']))

    @classmethod
    def add_failing_model_service(cls):
        service_config = PredictionServiceConfig(name='failing-model',
            version='B', model=None, meta={'1': 'one', 'two': 2})
        cls.model_app.add_service(service_config)


if __name__ == '__main__':
    unittest.main()
