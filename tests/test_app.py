"""
Tests for the `.app` attribute belonging to an instance of `porter.ModelService`.
"""


import json
import re
import warnings

import unittest
from unittest import mock

import flask
from werkzeug import exceptions as exc
from porter import __version__
from porter import constants as cn
from porter.datascience import BaseModel, BasePostProcessor, BasePreProcessor
from porter.services import ModelApp, BaseService, PredictionService
import porter.schemas as sc


@mock.patch('porter.responses.api.request_id', lambda: '123')
class TestAppPredictions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # define objects for model 1
        class Preprocessor1(BasePreProcessor):
            def process(self, X):
                X = X.copy() # silence SettingWithCopyWarning
                X['feature2'] = X.feature2.astype(str)
                return X
        class Model1(BaseModel):
            feature2_map = {str(x+1): x for x in range(5)}
            def predict(self, X):
                return X['feature1'] * X.feature2.map(self.feature2_map)
        class Postprocessor1(BasePostProcessor):
            def process(self, X_input, X_preprocessed, predictions):
                return predictions * -1
        feature_schema1 = sc.Object(
            properties={
                'feature1': sc.Number(), 
                'feature2': sc.Number(),
            }
        )

        # define objects for model 2
        class Preprocessor2(BasePreProcessor):
            def process(self, X):
                X['feature3'] = range(len(X))
                return X
        class Model2(BaseModel):
            def predict(self, X):
                return X['feature1'] + X['feature3']
        feature_schema2 = sc.Object(properties={'feature1': sc.Number()})
        def user_check(X):
            if (X.feature1 == 0).any():
                raise exc.UnprocessableEntity

        # define objects for model 3
        class Model3(BaseModel):
            def predict(self, X):
                return X['feature1'] * -1
        feature_schema3 = sc.Object(properties={'feature1': sc.Number()})
        wrong_prediction_schema3 = sc.Number(additional_params=dict(minimum=0))

        cls.prediction_service_error = E = Exception('this mock service failed during prediction')
        class ModelFailing(BaseModel):
            def predict(self, X):
                raise E

        # define configs and add services to app
        prediction_service1 = PredictionService(
            model=Model1(),
            name='a-model',
            api_version='v0',
            action='predict',
            preprocessor=Preprocessor1(),
            postprocessor=Postprocessor1(),
            feature_schema=feature_schema1,
            validate_request_data=True,
            batch_prediction=True
        )
        prediction_service2 = PredictionService(
            model=Model2(),
            name='anotherModel',
            api_version='v1',
            namespace='n/s/',
            preprocessor=Preprocessor2(),
            postprocessor=None,
            feature_schema=feature_schema2,
            validate_request_data=True,
            batch_prediction=True,
            additional_checks=user_check
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            prediction_service3 = PredictionService(
                model=Model3(),
                name='model-3',
                api_version='v0.0-alpha',
                preprocessor=None,
                postprocessor=None,
                feature_schema=feature_schema3,
                validate_request_data=True,
                validate_response_data=True,
                batch_prediction=False,
                meta={'algorithm': 'randomforest', 'lasttrained': 1}
            )
            prediction_service4 = PredictionService(
                model=Model3(),
                name='model-4',
                api_version='v0.0-alpha',
                preprocessor=None,
                postprocessor=None,
                feature_schema=feature_schema3,
                validate_request_data=True,
                validate_response_data=True,
                batch_prediction=False,
                meta={'algorithm': 'randomforest', 'lasttrained': 1}
            )
            prediction_service5 = PredictionService(
                model=Model3(),
                name='model-5',
                api_version='v0.0-alpha',
                preprocessor=None,
                postprocessor=None,
                feature_schema=feature_schema3,
                prediction_schema=wrong_prediction_schema3,
                validate_request_data=True,
                validate_response_data=True,
                batch_prediction=False,
                meta={'algorithm': 'randomforest', 'lasttrained': 1}
            )
        prediction_service_failing = PredictionService(
            model=ModelFailing(),
            name='failing-model',
            api_version='v1',
            action='fail',
        )
        cls.model_app = ModelApp([
            prediction_service1,
            prediction_service2,
            prediction_service3,
            prediction_service4,
            prediction_service5,
            prediction_service_failing,
        ])
        cls.app = cls.model_app.app.test_client()

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
        actual1 = self.app.post('/a-model/v0/predict', data=json.dumps(post_data1))
        actual1 = json.loads(actual1.data)
        actual2 = self.app.post('/n/s/anotherModel/v1/prediction', data=json.dumps(post_data2))
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

    def test_prediction_bad_requests_400(self):
        actual = self.app.post('/a-model/v0/predict', data='cannot be parsed')
        self.assertTrue(actual.status_code, 400)
        expected_data = {
            'request_id': 123,
            'error': {
                'name': 'BadRequest',
                'messages': ['The browser (or proxy) sent a request that this server could not understand.'],
            }
        }
        actual_data = json.loads(actual.data)
        self.assertIn('request_id', actual_data)
        self.assertEqual(expected_data['error'], actual_data['error'])

    def test_asdkfj(self):
        # TODO: rename
        post_data2 = [{'id': 1, 'feature1': 2}, {'id': 2, 'feature1': 2}]
        resp = self.app.post('/model-3/v0.0-alpha/prediction', data=json.dumps(post_data2))
        print(resp.json)
        self.assertEqual(resp.status_code, 422)

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
                      {'id': 1, 'feature1': 1}]
        # contains 0 values that don't pass user check
        post_data6 = [{'id': 1, 'feature1': 1, 'feature2': 1},
                      {'id': 1, 'feature1': 0, 'feature2': 1}]
        actuals = [
            self.app.post('/a-model/v0/predict', data=json.dumps(post_data1)),
            self.app.post('/model-3/v0.0-alpha/prediction', data=json.dumps(post_data2)),
            self.app.post('/n/s/anotherModel/v1/prediction', data=json.dumps(post_data3)),
            self.app.post('/model-3/v0.0-alpha/prediction', data=json.dumps(post_data4)),
            self.app.post('/a-model/v0/predict', data=json.dumps(post_data5)),
            self.app.post('/n/s/anotherModel/v1/prediction', data=json.dumps(post_data6)),
        ]
        # check status codes
        self.assertTrue(all(actual.status_code == 422 for actual in actuals))
        # check that all objects have error key
        self.assertTrue(all('error' in json.loads(actual.data) for actual in actuals))
        # check response values
        expected_error_values = [
            {'name': 'UnprocessableEntity'},
            {'name': 'UnprocessableEntity'},
            {'name': 'UnprocessableEntity'},
            {'name': 'UnprocessableEntity'},
            {'name': 'UnprocessableEntity'},
            {'name': 'UnprocessableEntity'},
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

    def test_prediction_response_valid_schema(self):
        # test that validation passes for valid response
        post_data4 = {'id': 1, 'feature1': 5}
        actual4 = self.app.post('/model-4/v0.0-alpha/prediction', data=json.dumps(post_data4))
        actual4 = json.loads(actual4.data)
        expected4 = {
            'request_id': '123',
            'model_context': {
                'model_name': 'model-4',
                'api_version': 'v0.0-alpha',
                'model_meta': {
                    'algorithm': 'randomforest',
                    'lasttrained': 1
                }
            },
            'predictions': {'id': 1, 'prediction': -5}
        }
        self.assertEqual(actual4, expected4)

    def test_prediction_response_invalid_schema(self):
        # test that validation fails for invalid response
        post_data5 = {'id': 1, 'feature1': 5}
        actual5 = self.app.post('/model-5/v0.0-alpha/prediction', data=json.dumps(post_data5))
        actual5 = json.loads(actual5.data)
        self.assertRegex(
            actual5['error']['messages'][0],
            'Schema validation failed: data.predictions.prediction must be bigger')

    def test_get_prediction_endpoints(self):
        resp1 = self.app.get('/a-model/v0/predict')
        resp2 = self.app.get('/model-3/v0.0-alpha/prediction')
        resp3 = self.app.get('/n/s/anotherModel/v1/prediction')
        self.assertEqual(resp1.status_code, 200)
        self.assertEqual(resp2.status_code, 200)
        self.assertEqual(resp3.status_code, 200)

    @mock.patch('porter.services.BaseService._log_error')
    def test_prediction_service_that_fails(self, mock__log_error):
        # test how unhandled exceptions are treated
        resp = self.app.post('/failing-model/v1/fail', data='{}')
        self.assertEqual(resp.status_code, 500)
        self.assertEqual(resp.json['error']['messages'], ['Could not serve model results successfully.'])
        # make sure the original exception was logged
        mock__log_error.assert_called_with(self.prediction_service_error)


class TestOpenAPIDocumentationDefaults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        service1 = PredictionService(
            name='service1',
            api_version='2',
            model=None,  # we're not going to make calls for predictions here
            feature_schema=sc.Object(properties={'a': sc.Integer(), 'b': sc.Integer(), 'c': sc.Number()})
        )
        service1 = PredictionService(
            namespace='ns',
            name='service2',
            api_version='1',
            model=None,  # we're not going to make calls for predictions here
            feature_schema=sc.Object(properties={'a': sc.Integer(), 'b': sc.Integer()})
        )

    def test_docs_url(self):
        model_app = ModelApp([], expose_docs=True)
        app = model_app.app.test_client()
        resp = app.get('/docs/')
        self.assertEqual(resp.status_code, 200)

    def test_can_serve_swagger_files(self):
        model_app = ModelApp([], expose_docs=True)
        app = model_app.app.test_client()
        files = [
            'favicon-16x16.png',
            'favicon-32x32.png',
            'index.html',
            'oauth2-redirect.html',
            'swagger-ui-bundle.js',
            'swagger-ui-bundle.js.map',
            'swagger-ui-standalone-preset.js',
            'swagger-ui-standalone-preset.js.map',
            'swagger-ui.css',
            'swagger-ui.css.map',
            'swagger-ui.js',
            'swagger-ui.js.map',
            'swagger_template.html',
        ]
        for filename in files:
            resp = app.get(f'/assets/swagger-ui/{filename}')
            self.assertEqual(resp.status_code, 200)

    def test_docs_prefix(self):
        model_app = ModelApp([], docs_prefix='/my/docs/ns', expose_docs=True)
        app = model_app.app.test_client()
        resp = app.get('/my/docs/ns/docs/')
        self.assertEqual(resp.status_code, 200)

        files = [
            'favicon-16x16.png',
            'favicon-32x32.png',
            'index.html',
            'oauth2-redirect.html',
            'swagger-ui-bundle.js',
            'swagger-ui-bundle.js.map',
            'swagger-ui-standalone-preset.js',
            'swagger-ui-standalone-preset.js.map',
            'swagger-ui.css',
            'swagger-ui.css.map',
            'swagger-ui.js',
            'swagger-ui.js.map',
            'swagger_template.html',
        ]
        for filename in files:
            resp = app.get(f'/my/docs/ns/assets/swagger-ui/{filename}')
            self.assertEqual(resp.status_code, 200)

    def test_docs_paths(self):
        class SC(BaseService):
            def serve(self): pass
            def status(self): pass
            action = 'endpoint'

        service = SC(name='the', api_version='v1')
        model_app = ModelApp([service], expose_docs=True)
        docs_json = model_app.docs_json
        expected_keys = ['openapi', 'info', 'paths', 'components']
        self.assertEqual(set(docs_json.keys()), set(expected_keys))
        expected_paths = ['/the/v1/endpoint', '/-/alive', '/-/ready']
        self.assertEqual(set(docs_json['paths'].keys()), set(expected_paths))


@mock.patch('porter.responses.api.request_id', lambda: '123')
class TestAppHealthChecks(unittest.TestCase):
    def test_liveness_live(self):
        model_app = ModelApp([])
        app = model_app.app.test_client()
        resp = app.get('/-/alive')
        self.assertEqual(resp.status_code, 200)

    def test_readiness_not_ready1(self):
        model_app = ModelApp([])
        app = model_app.app.test_client()
        resp_alive = app.get('/-/alive')
        resp_ready = app.get('/-/ready')
        expected_data = {
            'request_id': '123',
            'porter_version': __version__,
            'deployed_on': cn.HEALTH_CHECK_VALUES.DEPLOYED_ON,
            'services': {},
            'app_meta': {
                'description': '<div></div><div><p>(porter v0.16.5)</p></div>',
                'expose_docs': False,
                'name': None,
                'version': None},
        }
        self.assertEqual(resp_alive.status_code, 200)
        self.assertEqual(resp_ready.status_code, 503)
        alive_response = json.loads(resp_alive.data)
        ready_respnose = json.loads(resp_ready.data)
        self.assertEqual(alive_response, expected_data)
        self.assertEqual(ready_respnose, expected_data)
        # make sure the defined schema matches reality
        sc.health_check.validate(alive_response)  # should not raise exception
        sc.health_check.validate(ready_respnose)  # should not raise exception

    @mock.patch('porter.services.PredictionService.__init__')
    def test_readiness_not_ready2(self, mock_init):
        mock_init.return_value = None
        class C(PredictionService):
            status = 'NOTREADY'
            request_schemas = {}
            response_schemas = {}
        svc = C()
        svc.name  = 'foo'
        svc.api_version = 'bar'
        svc.meta = {'k': 1}
        svc.id = 'foo:bar'
        svc.endpoint = '/foo/bar/'
        svc.route_kwargs = {}

        model_app = ModelApp([svc])
        app = model_app.app.test_client()

        resp_alive = app.get('/-/alive')
        resp_ready = app.get('/-/ready')
        expected_data = {
            'request_id': '123',
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
            'app_meta': {
                'description': '<div></div><div><p>(porter v0.16.5)</p></div>',
                'expose_docs': False,
                'name': None,
                'version': None},
        }
        self.assertEqual(resp_alive.status_code, 200)
        self.assertEqual(resp_ready.status_code, 503)
        alive_response = json.loads(resp_alive.data)
        ready_respnose = json.loads(resp_ready.data)
        self.assertEqual(alive_response, expected_data)
        self.assertEqual(ready_respnose, expected_data)
        # make sure the defined schema matches reality
        sc.health_check.validate(alive_response)  # should not raise exception
        sc.health_check.validate(ready_respnose)  # should not raise exception

    @mock.patch('porter.services.PredictionService.__init__')
    def test_readiness_ready_ready1(self, mock_init):
        mock_init.return_value = None
        svc = PredictionService()
        svc.name = 'model1'
        svc.api_version = '1.0.0'
        svc.id = 'model1'
        svc.endpoint = '/model1/1.0.0/prediction'
        svc.meta = {'foo': 1, 'bar': 2}
        svc.response_schemas = {}
        svc.request_schemas = {}

        model_app = ModelApp([svc])
        app = model_app.app.test_client()

        resp_alive = app.get('/-/alive')
        resp_ready = app.get('/-/ready')
        expected_data = {
            'request_id': '123',
            'porter_version': __version__,
            'deployed_on': cn.HEALTH_CHECK_VALUES.DEPLOYED_ON,
            'app_meta': {
                'description': '<div></div><div><p>(porter v0.16.5)</p></div>',
                'expose_docs': False,
                'name': None,
                'version': None},
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
        alive_response = json.loads(resp_alive.data)
        ready_respnose = json.loads(resp_ready.data)
        self.assertEqual(alive_response, expected_data)
        self.assertEqual(ready_respnose, expected_data)
        # make sure the defined schema matches reality
        sc.health_check.validate(alive_response)  # should not raise exception
        sc.health_check.validate(ready_respnose)  # should not raise exception

    @mock.patch('porter.services.PredictionService.__init__')
    def test_readiness_ready_ready2(self, mock_init):
        mock_init.return_value = None
        svc1 = PredictionService()
        svc1.name = 'model1'
        svc1.api_version = '1.0.0'
        svc1.id = 'model1:1.0.0'
        svc1.endpoint = '/model1/1.0.0/prediction'
        svc1.meta = {'foo': 1, 'bar': 2}
        svc1.response_schemas = {}
        svc1.request_schemas = {}        
        svc2 = PredictionService()
        svc2.name = 'model2'
        svc2.api_version = 'v0'
        svc2.id = 'model2:v0'
        svc2.endpoint = '/model2/v0/prediction'
        svc2.meta = {'foo': 1}
        svc2.response_schemas = {}
        svc2.request_schemas = {}  

        model_app = ModelApp([svc1, svc2])
        app = model_app.app.test_client()

        resp_alive = app.get('/-/alive')
        resp_ready = app.get('/-/ready')
        expected_data = {
            'request_id': '123',
            'porter_version': __version__,
            'deployed_on': cn.HEALTH_CHECK_VALUES.DEPLOYED_ON,
            'app_meta': {
                'description': '<div></div><div><p>(porter v0.16.5)</p></div>',
                'expose_docs': False,
                'name': None,
                'version': None},
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
        alive_response = json.loads(resp_alive.data)
        ready_respnose = json.loads(resp_ready.data)
        self.assertEqual(alive_response, expected_data)
        self.assertEqual(ready_respnose, expected_data)
        # make sure the defined schema matches reality
        sc.health_check.validate(alive_response)  # should not raise exception
        sc.health_check.validate(ready_respnose)  # should not raise exception

    def test_root(self):
        model_app = ModelApp([])
        app = model_app.app.test_client()
        resp = app.get('/')
        self.assertEqual(resp.status_code, 200)

        model_app = ModelApp([], expose_docs=True)
        app = model_app.app.test_client()
        resp = app.get('/')
        self.assertEqual(resp.status_code, 302)


@mock.patch('porter.services.porter_responses.api.request_id', lambda: 123)
@mock.patch('porter.services.cf.return_message_on_error', True)
@mock.patch('porter.services.cf.return_traceback_on_error', True)
@mock.patch('porter.services.cf.return_user_data_on_error', True)
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

        prediction_service = PredictionService(name='failing-model',
            api_version='B', model=None, meta={'1': 'one', 'two': 2})

        cls.model_app = ModelApp([prediction_service])
        flask_app = cls.model_app.app
        @flask_app.route('/test-error-handling/', methods=['POST'])
        def test_error():
            flask.request.get_json(force=True)
            raise Exception('exceptional testing of exceptions')
        cls.app_test_client = flask_app.test_client()

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
                'messages': ['The requested URL was not found on the server. '
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
                'name': 'InternalServerError',
                'messages': ['Could not serve model results successfully.'],
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


@mock.patch('porter.services.porter_responses.api.request_id', lambda: 123)
@mock.patch('porter.services.cf.return_message_on_error', True)
@mock.patch('porter.services.cf.return_traceback_on_error', False)
@mock.patch('porter.services.cf.return_user_data_on_error', False)
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

        prediction_service = PredictionService(name='failing-model',
            api_version='B', model=None, meta={'1': 'one', 'two': 2})

        cls.model_app = ModelApp([prediction_service])

        flask_app = cls.model_app.app
        @flask_app.route('/test-error-handling/', methods=['POST'])
        def test_error():
            flask.request.get_json(force=True)
            raise Exception('exceptional testing of exceptions')
        cls.app_test_client = flask_app.test_client()

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
                'name': 'InternalServerError',
                'messages': ['Could not serve model results successfully.'],
            }
        }
        self.assertEqual(resp.status_code, 500)
        self.assertEqual(actual['model_context'], expected['model_context'])
        self.assertEqual(actual['model_context']['model_meta'], expected['model_context']['model_meta'])
        self.assertEqual(actual['error'], expected['error'])
        self.assertEqual(actual['request_id'], expected['request_id'])
        self.assertEqual(actual['error'], expected['error'])


if __name__ == '__main__':
    unittest.main()
