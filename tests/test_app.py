"""
Tests for the `.app` attribute belonging to an instance of `porter.ModelService`.
"""


import json
import unittest
from unittest import mock

import flask
from porter.constants import KEYS
from porter.datascience import BaseModel, BaseProcessor
from porter.services import ModelApp, PredictionServiceConfig

MODEL_NAME = KEYS.PREDICTION.MODEL_NAME
MODEL_VERSION = KEYS.PREDICTION.MODEL_VERSION
PREDICTIONS = KEYS.PREDICTION.PREDICTIONS
ID = KEYS.PREDICTION.ID
PREDICTION = KEYS.PREDICTION.PREDICTION


class TestAppPredictions(unittest.TestCase):
    def setUp(self):
        self.model_app = ModelApp()
        self.app = self.model_app.app.test_client()

    def test(self):
        # define objects for model 1
        class Preprocessor1(BaseProcessor):
            def process(self, X):
                X['feature2'] = X.feature2.astype(str)
                return X
        class Model1(BaseModel):
            feature2_map = {str(x+1): x for x in range(5)}
            def predict(self, X):
                return X['feature1'] * X.feature2.map(self.feature2_map)
        class Postprocessor1(BaseProcessor):
            def process(self, X):
                return X * -1
        input_features1 = ['feature1', 'feature2']
        post_data1 = [
            {ID: 1, 'feature1': 2, 'feature2': 1},
            {ID: 2, 'feature1': 2, 'feature2': 2},
            {ID: 3, 'feature1': 2, 'feature2': 3},
            {ID: 4, 'feature1': 2, 'feature2': 4},
            {ID: 5, 'feature1': 2, 'feature2': 5},
        ]

        # define objects for model 2
        class Preprocessor2(BaseProcessor):
            def process(self, X):
                X['feature3'] = range(len(X))
                return X
        class Model2(BaseModel):
            def predict(self, X):
                return X['feature1'] + X['feature3']
        input_features2 = ['feature1']
        post_data2 = [
            {ID: 1, 'feature1': 10},
            {ID: 2, 'feature1': 10},
            {ID: 3, 'feature1':  1},
            {ID: 4, 'feature1':  3},
            {ID: 5, 'feature1':  3},
        ]

        # define objects for model 3
        class Model3(BaseModel):
            def predict(self, X):
                return X['feature1'] * -1
        input_features3 = ['feature1']
        post_data3 = {ID: 1, 'feature1': 5}

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
            batch_prediction=True
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
        self.model_app.add_service(service_config1)
        self.model_app.add_service(service_config2)
        self.model_app.add_service(service_config3)

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
            'predictions': [{'id': 1, 'prediction': -5}]
        }
        self.assertEqual(actual1, expected1)
        self.assertEqual(actual2, expected2)
        self.assertEqual(actual3, expected3)


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
        self.model_app.add_service(cf)
        resp_alive = self.app.get('/-/alive')
        resp_ready = self.app.get('/-/ready')
        expected_data = {
            'services': {
                'model1': {
                    'status': 'READY',
                    'version': '1.0.0',
                    'endpoint': '/model1/prediction',
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
        cf2 = PredictionServiceConfig()
        cf2.name = 'model2'
        cf2.version = '0.0.0'
        cf2.id = 'model2:0.0.0'
        cf2.endpoint = '/model2/prediction'
        self.model_app.add_services(cf1, cf2)
        resp_alive = self.app.get('/-/alive')
        resp_ready = self.app.get('/-/ready')
        expected_data = {
            'services': {
                'model1:1.0.0': {
                    'status': 'READY',
                    'version': '1.0.0',
                    'endpoint': '/model1/prediction'
                },
                'model2:0.0.0': {
                    'status': 'READY',
                    'version': '0.0.0',
                    'endpoint': '/model2/prediction'
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
        app = ModelApp().app
        @app.route('/test-error-handling/', methods=['POST'])
        def test_error():
            flask.request.get_json(force=True)
            raise Exception('exceptional testing of exceptions')
        cls.app = app.test_client()

    def test_bad_request(self):
        # note data is unreadable JSON, thus a BadRequest
        resp = self.app.post('/test-error-handling/', data='')
        self.validate_error_response(resp, 'BadRequest', 400)

    def test_not_found(self):
        resp = self.app.get('/not-found/')
        self.validate_error_response(resp, 'NotFound', 404)

    def test_method_not_allowed(self):
        resp = self.app.get('/test-error-handling/')
        self.validate_error_response(resp, 'MethodNotAllowed', 405)

    def test_internal_server_error(self):
        resp = self.app.post('/test-error-handling/', data='{}')
        self.validate_error_response(resp, 'Exception', 500, 'exceptional', 'raise Exception')

    def validate_error_response(self, response, error, status_code, message_substr=None,
                                traceback_substr=None):
        data = json.loads(response.data)
        self.assertEqual(data['error'], error)
        self.assertEqual(response.status_code, status_code)
        # even if we don't check the values of message or traceback at least
        # make sure that these keys are returned to the user.
        self.assertIn('message', data)
        self.assertIn('traceback', data)
        if not message_substr is None:
            if isinstance(data['message'], list):
                # if error is a builtin, it does not have a description. Thus
                # data['message'] is Exception().args, i.e. a list once jsonified
                self.assertTrue(any(message_substr in arg for arg in data['message']))
            else:
                self.assertIn(message_substr, data['message'])
        if not traceback_substr is None:
            self.assertIn(traceback_substr, data['traceback'])


if __name__ == '__main__':
    unittest.main()
