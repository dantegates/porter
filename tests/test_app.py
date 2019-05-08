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
from porter.services import ModelApp, PredictionService, MiddlewareService
from porter import constants as cn


@mock.patch('porter.responses.api.request_id', lambda: 123)
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
        prediction_service1 = PredictionService(
            model=Model1(),
            name='a-model',
            api_version='v0',
            preprocessor=Preprocessor1(),
            postprocessor=Postprocessor1(),
            input_features=input_features1,
            allow_nulls=False,
            batch_prediction=True
        )
        prediction_service2 = PredictionService(
            model=Model2(),
            name='anotherModel',
            api_version='v1',
            preprocessor=Preprocessor2(),
            postprocessor=None,
            input_features=input_features2,
            allow_nulls=False,
            batch_prediction=True,
            additional_checks=user_check
        )
        prediction_service3 = PredictionService(
            model=Model3(),
            name='model-3',
            api_version='v0.0-alpha',
            preprocessor=None,
            postprocessor=None,
            input_features=input_features3,
            allow_nulls=False,
            batch_prediction=False,
            meta={'algorithm': 'randomforest', 'lasttrained': 1}
        )
        with mock.patch('porter.services.api', **{'validate_url.return_value': True}):
            middleware_service = MiddlewareService(
                name='model-3',
                api_version='version1',
                model_endpoint=prediction_service3.endpoint,
                max_workers=None)
        cls.model_app.add_service(prediction_service1)
        cls.model_app.add_service(prediction_service2)
        cls.model_app.add_service(prediction_service3)
        cls.model_app.add_service(middleware_service)

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
        actual1 = self.app.post('/a-model/v0/prediction', data=json.dumps(post_data1))
        actual1 = json.loads(actual1.data)
        actual2 = self.app.post('/anotherModel/v1/prediction', data=json.dumps(post_data2))
        actual2 = json.loads(actual2.data)
        actual3 = self.app.post('/model-3/v0.0-alpha/prediction', data=json.dumps(post_data3))
        actual3 = json.loads(actual3.data)
        expected1 = {
            'request_id': 0,
            'model_context': {
                'model_name': 'a-model',
                'api_version': 'v0',
                'model_meta': {}
            },
            'predictions': [
                {'id': 1, 'prediction': 0},
                {'id': 2, 'prediction': -2},
                {'id': 3, 'prediction': -4},
                {'id': 4, 'prediction': -6},
                {'id': 5, 'prediction': -8},
            ]
        }
        expected2 = {
            'request_id': 1,
            'model_context': {
                'model_name': 'anotherModel',
                'api_version': 'v1',
                'model_meta': {}
            },
            'predictions': [
                {'id': 1, 'prediction': 10},
                {'id': 2, 'prediction': 11},
                {'id': 3, 'prediction': 3},
                {'id': 4, 'prediction': 6},
                {'id': 5, 'prediction': 7},
            ]
        }
        expected3 = {
            'request_id': 123,
            'model_context': {
                'model_name': 'model-3',
                'api_version': 'v0.0-alpha',
                'model_meta': {
                    'algorithm': 'randomforest',
                    'lasttrained': 1
                }
            },
            'predictions': {'id': 1, 'prediction': -5}
        }
        for key in ['model_context', 'predictions']:
            self.assertCountEqual(actual1[key], expected1[key])
            self.assertCountEqual(actual2[key], expected2[key])
            self.assertCountEqual(actual3[key], expected3[key])

    @mock.patch('porter.services.api.post')
    def test_middleware(self, mock_post):
        # test middleware
        def post(url, data, **kwargs):
            response = self.app.post(url, data=json.dumps(data))
            m = mock.Mock()
            m.json.side_effect = lambda: json.loads(response.data)
            return m
        mock_post.side_effect = post

        # only the third service supports instance predictions
        post_data = [{'id': i, 'feature1': i*2} for i in range(10)]
        actual = self.app.post('/model-3/version1/batchPrediction', data=json.dumps(post_data))
        actual = json.loads(actual.data)
        expected = [
            {
                'model_context': {
                    'model_name': 'model-3',
                    'api_version': 'v0.0-alpha',
                    'model_meta': {
                        'algorithm': 'randomforest',
                        'lasttrained': 1
                    }
                },
                'predictions': {'id': d['id'], 'prediction': -d['feature1']}
            }
            for d in post_data
        ]
        actual_hashable = [sorted(tuple(x.items())) for x in actual]
        expected_hashable = [sorted(tuple(x.items())) for x in expected]
        self.assertCountEqual(actual_hashable, expected_hashable)

    @mock.patch('porter.responses.cf.return_request_id_on_error', True)
    def test_prediction_bad_requests_400(self):
        actual = self.app.post('/a-model/v0/prediction', data='cannot be parsed')
        self.assertTrue(actual.status_code, 400)
        expectd_data = {
            'request_id': 123,
            'error': {
                'name': 'BadRequest',
                'messages': ['The browser (or proxy) sent a request that this server could not understand.'],
            }
        }
        actual_data = json.loads(actual.data)
        self.assertIn('request_id', actual_data)
        self.assertEqual(expectd_data['error'], actual_data['error'])

    def test_prediction_bad_requests_422(self):
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
            self.app.post('/a-model/v0/prediction', data=json.dumps(post_data1)),
            self.app.post('/model-3/v0.0-alpha/prediction', data=json.dumps(post_data2)),
            self.app.post('/anotherModel/v1/prediction', data=json.dumps(post_data3)),
            self.app.post('/model-3/v0.0-alpha/prediction', data=json.dumps(post_data4)),
            self.app.post('/a-model/v0/prediction', data=json.dumps(post_data5)),
            self.app.post('/anotherModel/v1/prediction', data=json.dumps(post_data6)),
        ]
        # check status codes
        self.assertTrue(all(actual.status_code == 422 for actual in actuals))
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
            {'name': 'BadRequest'},
        ]
        for actual, expectations in zip(actuals, expected_error_values):
            actual_error_obj = json.loads(actual.data)['error']
            for key, value in expectations.items():
                self.assertEqual(actual_error_obj[key], value)
        # check that model context data is passed into responses
        expected_model_context_values = [
            {'model_name': 'a-model', 'api_version': 'v0'},
            {'model_name': 'model-3', 'api_version': 'v0.0-alpha',
             'model_meta': {'algorithm': 'randomforest', 'lasttrained': 1}},
            {'model_name': 'anotherModel', 'api_version': 'v1'},
            {'model_name': 'model-3', 'api_version': 'v0.0-alpha',
             'model_meta': {'algorithm': 'randomforest', 'lasttrained': 1}},
            {'model_name': 'a-model', 'api_version': 'v0'},
            {'model_name': 'anotherModel', 'api_version': 'v1'},
        ]
        for actual, expectations in zip(actuals, expected_model_context_values):
            actual_error_obj = json.loads(actual.data)
            for key, value in expectations.items():
                self.assertEqual(actual_error_obj['model_context'][key], value)

    def test_get_prediction_endpoints(self):
        resp1 = self.app.get('/a-model/v0/prediction')
        resp2 = self.app.get('/model-3/v0.0-alpha/prediction')
        resp3 = self.app.get('/anotherModel/v1/prediction')
        self.assertEqual(resp1.status_code, 200)
        self.assertEqual(resp2.status_code, 200)
        self.assertEqual(resp3.status_code, 200)

class TestAppHealthChecks(unittest.TestCase):
    def setUp(self):
        self.model_app = ModelApp()
        self.app = self.model_app.app.test_client()

    def tearDown(self):
        self.model_app._services = []

    def test_liveness_live(self):
        resp = self.app.get('/-/alive')
        self.assertEqual(resp.status_code, 200)

    def test_readiness_not_ready1(self):
        resp_alive = self.app.get('/-/alive')
        resp_ready = self.app.get('/-/ready')
        expected_data = {
            'porter_version': __version__,
            'deployed_on': cn.HEALTH_CHECK_VALUES.DEPLOYED_ON,
            'services': {},
            'app_meta': {}
        }
        self.assertEqual(resp_alive.status_code, 200)
        self.assertEqual(resp_ready.status_code, 503)
        self.assertEqual(json.loads(resp_alive.data), expected_data)
        self.assertEqual(json.loads(resp_ready.data), expected_data)

    @mock.patch('porter.services.PredictionService.__init__')
    @mock.patch('porter.services.api.App')
    def test_readiness_not_ready2(self, mock_App, mock_init):
        mock_init.return_value = None
        class C(PredictionService):
            status = 'NOTREADY'
        cf = C()
        cf.name  = 'foo'
        cf.api_version = 'bar'
        cf.meta = {'k': 1}
        cf.id = 'foo:bar'
        cf.endpoint = '/foo/bar/'
        cf.route_kwargs = {}
        self.model_app.add_service(cf)
        resp_alive = self.app.get('/-/alive')
        resp_ready = self.app.get('/-/ready')
        expected_data = {
            'porter_version': __version__,
            'deployed_on': cn.HEALTH_CHECK_VALUES.DEPLOYED_ON,
            'services': {
                'foo:bar': {
                    'status': 'NOTREADY',
                    'endpoint': '/foo/bar/',
                    'model_context': {
                        'model_name': 'foo',
                        'api_version': 'bar',
                        'model_meta': {'k': 1}
                    }
                }
            },
            'app_meta': {}
        }
        self.assertEqual(resp_alive.status_code, 200)
        self.assertEqual(resp_ready.status_code, 503)
        self.assertEqual(json.loads(resp_alive.data), expected_data)
        self.assertEqual(json.loads(resp_ready.data), expected_data)

    @mock.patch('porter.services.PredictionService.__init__')
    @mock.patch('porter.services.api.App')
    def test_readiness_ready_ready1(self, mock_App, mock_init):
        mock_init.return_value = None
        cf = PredictionService()
        cf.name = 'model1'
        cf.api_version = '1.0.0'
        cf.id = 'model1'
        cf.endpoint = '/model1/1.0.0/prediction'
        cf.meta = {'foo': 1, 'bar': 2}
        self.model_app.add_service(cf)
        resp_alive = self.app.get('/-/alive')
        resp_ready = self.app.get('/-/ready')
        expected_data = {
            'porter_version': __version__,
            'deployed_on': cn.HEALTH_CHECK_VALUES.DEPLOYED_ON,
            'app_meta': {},
            'services': {
                'model1': {
                    'status': 'READY',
                    'endpoint': '/model1/1.0.0/prediction',
                    'model_context': {
                        'model_name': 'model1',
                        'api_version': '1.0.0',
                        'model_meta': {'foo': 1, 'bar': 2}
                    }
                }
            }
        }
        self.assertEqual(resp_alive.status_code, 200)
        self.assertEqual(resp_ready.status_code, 200)
        self.assertEqual(json.loads(resp_alive.data), expected_data)
        self.assertEqual(json.loads(resp_ready.data), expected_data)

    @mock.patch('porter.services.PredictionService.__init__')
    @mock.patch('porter.services.api.App')
    def test_readiness_ready_ready2(self, mock_App, mock_init):
        mock_init.return_value = None
        cf1 = PredictionService()
        cf1.name = 'model1'
        cf1.api_version = '1.0.0'
        cf1.id = 'model1:1.0.0'
        cf1.endpoint = '/model1/1.0.0/prediction'
        cf1.meta = {'foo': 1, 'bar': 2}
        cf2 = PredictionService()
        cf2.name = 'model2'
        cf2.api_version = 'v0'
        cf2.id = 'model2:v0'
        cf2.endpoint = '/model2/v0/prediction'
        cf2.meta = {'foo': 1}
        self.model_app.add_services(cf1, cf2)
        resp_alive = self.app.get('/-/alive')
        resp_ready = self.app.get('/-/ready')
        expected_data = {
            'porter_version': __version__,
            'deployed_on': cn.HEALTH_CHECK_VALUES.DEPLOYED_ON,
            'app_meta': {},
            'services': {
                'model1:1.0.0': {
                    'status': 'READY',
                    'endpoint': '/model1/1.0.0/prediction',
                    'model_context': {
                        'model_name': 'model1',
                        'api_version': '1.0.0',
                        'model_meta': {'foo': 1, 'bar': 2},
                    }
                },
                'model2:v0': {
                    'status': 'READY',
                    'endpoint': '/model2/v0/prediction',
                    'model_context': {
                        'model_name': 'model2',
                        'api_version': 'v0',
                        'model_meta': {'foo': 1},
                    }
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


@mock.patch('porter.services.porter_responses.api.request_id', lambda: 123)
@mock.patch('porter.services.cf.return_message_on_error', True)
@mock.patch('porter.services.cf.return_traceback_on_error', True)
@mock.patch('porter.services.cf.return_user_data_on_error', True)
@mock.patch('porter.services.cf.return_request_id_on_error', True)
class TestAppErrorHandling(unittest.TestCase):
    @classmethod
    @mock.patch('porter.services.BaseService._ids', set())
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
            'request_id': 123,
            'error': {
                'name': 'BadRequest',
                'messages': ['The browser (or proxy) sent a request that this server could not understand.'],
                # user_data is None when not passed or unreadable
                'user_data': None,
                'traceback': re.compile(r'.*raise\sBadRequest.*')
            }
        }
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(actual['error']['name'], expected['error']['name'])
        self.assertEqual(actual['request_id'], expected['request_id'])
        self.assertEqual(actual['error']['messages'], expected['error']['messages'])
        self.assertEqual(actual['error']['user_data'], expected['error']['user_data'])
        self.assertTrue(expected['error']['traceback'].search(actual['error']['traceback']))

    def test_not_found(self):
        resp = self.app_test_client.get('/not-found/')
        actual = json.loads(resp.data)
        expected = {
            'request_id': 123,
            'error': {
                'name': 'NotFound',
                'messages': ['The requested URL was not found on the server.  '
                             'If you entered the URL manually please check your spelling and '
                             'try again.'],
                'user_data': None,
                'traceback': re.compile(r'.*raise\sNotFound.*')
            }
        }
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(actual['error']['name'], expected['error']['name'])
        self.assertEqual(actual['request_id'], expected['request_id'])
        self.assertEqual(actual['error']['messages'], expected['error']['messages'])
        self.assertEqual(actual['error']['user_data'], expected['error']['user_data'])
        self.assertTrue(expected['error']['traceback'].search(actual['error']['traceback']))

    def test_method_not_allowed(self):
        resp = self.app_test_client.get('/test-error-handling/')
        actual = json.loads(resp.data)
        expected = {
            'request_id': 123,
            'error': {
                'name': 'MethodNotAllowed',
                'messages': ['The method is not allowed for the requested URL.'],
                'user_data': None,
                'traceback': re.compile(r'.*raise\sMethodNotAllowed.*')
            }
        }
        self.assertEqual(resp.status_code, 405)
        self.assertEqual(actual['error']['name'], expected['error']['name'])
        self.assertEqual(actual['request_id'], expected['request_id'])
        self.assertEqual(actual['error']['messages'], expected['error']['messages'])
        self.assertEqual(actual['error']['user_data'], expected['error']['user_data'])
        self.assertTrue(expected['error']['traceback'].search(actual['error']['traceback']))

    def test_internal_server_error(self):
        user_data = {"valid": "json"}
        resp = self.app_test_client.post('/test-error-handling/', data=json.dumps(user_data))
        actual = json.loads(resp.data)
        expected = {
            'request_id': 123,
            'error': {
                'name': 'Exception',
                'messages': ['exceptional testing of exceptions'],
                'user_data': user_data,
                'traceback': re.compile(r'.*raise\sException')
            }
        }
        self.assertEqual(resp.status_code, 500)
        self.assertEqual(actual['error']['name'], expected['error']['name'])
        self.assertEqual(actual['request_id'], expected['request_id'])
        self.assertEqual(actual['error']['messages'], expected['error']['messages'])
        self.assertEqual(actual['error']['user_data'], expected['error']['user_data'])
        self.assertTrue(expected['error']['traceback'].search(actual['error']['traceback']))

    @mock.patch('porter.services.PredictionService._predict')
    def test_prediction_fails(self, mock__predict):
        mock__predict.side_effect = Exception('testing a failing model')
        user_data = {'some test': 'data'}
        resp = self.app_test_client.post('/failing-model/B/prediction', data=json.dumps(user_data))
        actual = json.loads(resp.data)
        expected = {
            'model_context': {
                'model_name': 'failing-model',
                'api_version': 'B',
                'model_meta': {
                    '1': 'one',
                    'two': 2
                },
            },
            'request_id': 123,
            'error': {
                'name': 'PredictionError',
                'messages': ['an error occurred during prediction'],
                'user_data': user_data,
                'traceback': re.compile(r".*testing\sa\sfailing\smodel.*"),
            }
        }
        self.assertEqual(resp.status_code, 500)
        self.assertEqual(actual['model_context']['model_name'], expected['model_context']['model_name'])
        self.assertEqual(actual['model_context']['api_version'], expected['model_context']['api_version'])
        self.assertEqual(actual['model_context']['model_meta']['1'], expected['model_context']['model_meta']['1'])
        self.assertEqual(actual['model_context']['model_meta']['two'], expected['model_context']['model_meta']['two'])
        self.assertEqual(actual['error']['name'], expected['error']['name'])
        self.assertEqual(actual['request_id'], expected['request_id'])
        self.assertEqual(actual['error']['messages'], expected['error']['messages'])
        self.assertEqual(actual['error']['user_data'], expected['error']['user_data'])
        self.assertTrue(expected['error']['traceback'].search(actual['error']['traceback']))

    @classmethod
    def add_failing_model_service(cls):
        prediction_service = PredictionService(name='failing-model',
            api_version='B', model=None, meta={'1': 'one', 'two': 2})
        cls.model_app.add_service(prediction_service)


@mock.patch('porter.services.porter_responses.api.request_id', lambda: 123)
@mock.patch('porter.services.cf.return_message_on_error', True)
@mock.patch('porter.services.cf.return_traceback_on_error', False)
@mock.patch('porter.services.cf.return_user_data_on_error', False)
@mock.patch('porter.responses.cf.return_request_id_on_error', True)
class TestAppErrorHandlingCustomKeys(unittest.TestCase):
    @classmethod
    @mock.patch('porter.services.BaseService._ids', set())
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

    @mock.patch('porter.services.PredictionService._predict')
    def test(self, mock__predict):
        mock__predict.side_effect = Exception('testing a failing model')
        user_data = {'some test': 'data'}
        resp = self.app_test_client.post('/failing-model/B/prediction', data=json.dumps(user_data))
        actual = json.loads(resp.data)
        expected = {
            'model_context': {
                'model_name': 'failing-model',
                'api_version': 'B',
                'model_meta': {
                    '1': 'one',
                    'two': 2
                }
            },
            'request_id': 123,
            'error': {
                'name': 'PredictionError',
                'messages': ['an error occurred during prediction'],
            }
        }
        self.assertEqual(resp.status_code, 500)
        self.assertEqual(actual['model_context'], expected['model_context'])
        self.assertEqual(actual['model_context']['model_meta'], expected['model_context']['model_meta'])
        self.assertEqual(actual['error'], expected['error'])
        self.assertEqual(actual['request_id'], expected['request_id'])
        self.assertEqual(actual['error'], expected['error'])

    @classmethod
    def add_failing_model_service(cls):
        prediction_service = PredictionService(name='failing-model',
            api_version='B', model=None, meta={'1': 'one', 'two': 2})
        cls.model_app.add_service(prediction_service)        


if __name__ == '__main__':
    unittest.main()
