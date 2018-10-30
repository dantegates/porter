import json
import unittest
from unittest import mock

import numpy as np
import pandas as pd
from porter import constants as cn
from porter import exceptions as exc
from porter.services import (BaseService, MiddlewareService, ModelApp,
                             PredictionService, StatefulRoute,
                             serve_error_message)


class TestFuntionsUnit(unittest.TestCase):
    @mock.patch('flask.request')
    @mock.patch('flask.jsonify')
    def test_serve_error_message_status_codes_arbitrary_error(self, mock_flask_request, mock_flask_jsonify):
        # if the current error does not have an error code make sure
        # the response gets a 500
        error = ValueError('an error message')
        actual = serve_error_message(error)
        actual_status_code = 500
        expected_status_code = 500
        self.assertEqual(actual_status_code, expected_status_code)

    @mock.patch('flask.request')
    @mock.patch('flask.jsonify')
    def test_serve_error_message_status_codes_werkzeug_error(self, mock_flask_request, mock_flask_jsonify):
        # make sure that workzeug error codes get passed on to response
        error = ValueError('an error message')
        error.code = 123
        actual = serve_error_message(error)
        actual_status_code = 123
        expected_status_code = 123
        self.assertEqual(actual_status_code, expected_status_code)


class TestStatefulRoute(unittest.TestCase):
    def test_naming(self):
        class A(StatefulRoute):
            pass
        actual1 = A().__name__
        expected1 = 'a_1'
        actual2 = A().__name__
        expected2 = 'a_2'
        actual3 = A().__name__
        expected3 = 'a_3'
        self.assertEqual(actual1, expected1)
        self.assertEqual(actual2, expected2)
        self.assertEqual(actual3, expected3)


class TestPredictionService(unittest.TestCase):
    @mock.patch('porter.services.api.request_json')
    @mock.patch('porter.services.porter_responses.api.jsonify', lambda payload: payload)
    @mock.patch('porter.services.api.request_method', lambda: 'POST')
    @mock.patch('porter.services.BaseService._ids', set())
    def test_serve_success_batch(self, mock_request_json):
        mock_request_json.return_value = [
            {'id': 1, 'feature1': 10, 'feature2': 0},
            {'id': 2, 'feature1': 11, 'feature2': 1},
            {'id': 3, 'feature1': 12, 'feature2': 2},
            {'id': 4, 'feature1': 13, 'feature2': 3},
            {'id': 5, 'feature1': 14, 'feature2': 3},
        ]
        mock_model = mock.Mock()
        test_model_name = 'model'
        test_model_version = '1.0.0'
        mock_preprocessor = mock.Mock()
        mock_postprocessor = mock.Mock()
        allow_nulls = False

        feature_values = {str(x): x for x in range(5)}
        mock_model.predict = lambda X: X['feature1'] + X['feature2'].map(feature_values) + X['feature3']
        def preprocess(X):
            X['feature2'] = X.feature2.astype(str)
            X['feature3'] = range(len(X))
            return X
        mock_preprocessor.process = preprocess
        def postprocess(X_in, X_pre, preds):
            return preds * 2
        mock_postprocessor.process = postprocess
        serve_prediction = PredictionService(
            model=mock_model,
            name=test_model_name,
            version=test_model_version,
            meta={'1': '2', '3': 4},
            preprocessor=mock_preprocessor,
            postprocessor=mock_postprocessor,
            allow_nulls=allow_nulls,
            batch_prediction=True,
            additional_checks=None
        )
        actual = serve_prediction()
        expected = {
            'model_name': test_model_name,
            'model_version': test_model_version,
            '1': '2',
            '3': 4,
            'predictions': [
                {'id': 1, 'prediction': 20},
                {'id': 2, 'prediction': 26},
                {'id': 3, 'prediction': 32},
                {'id': 4, 'prediction': 38},
                {'id': 5, 'prediction': 42},
            ]
        }
        self.assertEqual(actual, expected)

    @mock.patch('porter.services.api.request_json')
    @mock.patch('porter.services.porter_responses.api.jsonify', lambda payload: payload)
    @mock.patch('porter.services.api.request_method', lambda: 'POST')
    @mock.patch('porter.services.BaseService._ids', set())
    def test_serve_success_single(self, mock_request_json):
        mock_request_json.return_value = {'id': 1, 'feature1': 10, 'feature2': 0}
        mock_model = mock.Mock()
        test_model_name = 'model'
        test_model_version = '1.0.0'
        mock_preprocessor = mock.Mock()
        mock_postprocessor = mock.Mock()
        allow_nulls = False

        feature_values = {str(x): x for x in range(5)}
        mock_model.predict = lambda X: X['feature1'] + X['feature2'].map(feature_values) + X['feature3']
        def preprocess(X):
            X['feature2'] = X.feature2.astype(str)
            X['feature3'] = range(len(X))
            return X
        mock_preprocessor.process = preprocess
        def postprocess(X_in, X_pre, preds):
            return preds * 2
        mock_postprocessor.process = postprocess
        serve_prediction = PredictionService(
            model=mock_model,
            name=test_model_name,
            version=test_model_version,
            meta={'1': '2', '3': 4},
            preprocessor=mock_preprocessor,
            postprocessor=mock_postprocessor,
            allow_nulls=allow_nulls,
            batch_prediction=False,
            additional_checks=None
        )
        actual = serve_prediction()
        expected = {
            'model_name': test_model_name,
            'model_version': test_model_version,
            '1': '2',
            '3': 4,
            'predictions': {'id': 1, 'prediction': 20}
        }
        self.assertEqual(actual, expected)

    @mock.patch('porter.services.PredictionService._predict')
    @mock.patch('porter.services.api')
    @mock.patch('porter.services.PredictionService.check_meta', lambda self, meta: meta)
    @mock.patch('porter.services.PredictionService.update_meta', lambda self, meta: meta)
    @mock.patch('porter.services.BaseService._ids', set())
    def test_serve_fail(self, mock_api, mock__predict):
        mock__predict.side_effect = Exception
        with self.assertRaises(exc.PredictionError):
            sp = PredictionService(
                model=mock.Mock(), name=mock.Mock(), version=mock.Mock(),
                meta=mock.Mock(), preprocessor=mock.Mock(), postprocessor=mock.Mock(),
                allow_nulls=mock.Mock(), batch_prediction=mock.Mock(),
                additional_checks=mock.Mock())
            sp()

    @mock.patch('flask.request')
    @mock.patch('flask.jsonify')
    def test_serve_with_processing_batch(self, mock_flask_jsonify, mock_flask_request):
        mock_model = mock.Mock()
        mock_flask_request.get_json.return_value = [{'id': None}]
        mock_model.predict.return_value = []
        mock_preprocessor = mock.Mock()
        mock_preprocessor.process.return_value = {}
        mock_postprocessor = mock.Mock()
        mock_postprocessor.process.return_value = []
        serve_prediction = PredictionService(
            model=mock_model,
            name=mock.Mock(),
            version=mock.Mock(),
            meta={},
            allow_nulls=mock.Mock(),
            preprocessor=mock_preprocessor,
            postprocessor=mock_postprocessor,
            batch_prediction=True,
            additional_checks=None
        )
        _ = serve_prediction()
        mock_preprocessor.process.assert_called()
        mock_postprocessor.process.assert_called()

    @mock.patch('flask.request')
    @mock.patch('flask.jsonify')
    @mock.patch('porter.services.BaseService._ids', set())
    def test_serve_no_processing_batch(self, mock_flask_jsonify, mock_flask_request):
        # make sure it doesn't break when processors are None
        model = model_name = model_version = allow_nulls = mock.Mock()
        mock_flask_request.get_json.return_value = [{'id': None}]
        model.predict.return_value = []
        serve_prediction = PredictionService(
            model=model,
            name=model_name,
            version=model_version,
            meta={},
            allow_nulls=allow_nulls,
            preprocessor=None,
            postprocessor=None,
            batch_prediction=True,
            additional_checks=None
        )
        _ = serve_prediction()

    @mock.patch('flask.request')
    @mock.patch('flask.jsonify')
    def test_serve_with_processing_single(self, mock_flask_jsonify, mock_flask_request):
        model = model_name = model_version = allow_nulls = mock.Mock()
        mock_flask_request.get_json.return_value = {'id': None}
        model.predict.return_value = [1]
        mock_preprocessor = mock.Mock()
        mock_preprocessor.process.return_value = {}
        mock_postprocessor = mock.Mock()
        mock_postprocessor.process.return_value = [1]
        serve_prediction = PredictionService(
            model=model,
            name=model_name,
            version=model_version,
            meta={},
            allow_nulls=allow_nulls,
            preprocessor=mock_preprocessor,
            postprocessor=mock_postprocessor,
            batch_prediction=False,
            additional_checks=None
        )
        _ = serve_prediction()
        mock_preprocessor.process.assert_called()
        mock_postprocessor.process.assert_called()

    @mock.patch('flask.request')
    @mock.patch('flask.jsonify')

    @mock.patch('porter.services.BaseService._ids', set())
    def test_serve_no_processing_single(self, mock_flask_jsonify, mock_flask_request):
        # make sure it doesn't break when processors are None
        model = model_name = model_version = allow_nulls = mock.Mock()
        mock_flask_request.get_json.return_value = {'id': None}
        model.predict.return_value = [1]
        serve_prediction = PredictionService(
            model=model,
            name=model_name,
            version=model_version,
            meta={},
            allow_nulls=allow_nulls,
            preprocessor=None,
            postprocessor=None,
            batch_prediction=False,
            additional_checks=None
        )
        _ = serve_prediction()

    def test_check_request_pass(self):
        # no error should be raised
        X = pd.DataFrame(
            [[0, 1, 2, 3], [4, 5, 6, 7]],
            columns=['id', 'one', 'two', 'three'])
        PredictionService.check_request(X, ['id', 'one', 'two', 'three'])

    def test_check_request_fail_missing_id(self):
        X = pd.DataFrame(
            [[0, 1, 2, 3], [4, 5, 6, 7]],
            columns=['missing', 'one', 'two', 'three'])
        with self.assertRaises(exc.RequestMissingFields):
            PredictionService.check_request(X, ['id', 'one', 'two', 'three'])

    def test_check_request_fail_missing_id_column(self):
        X = pd.DataFrame(
            [[0, 1, 2, 3], [4, 5, 6, 7]],
            columns=['missing', 'one', 'two', 'three'])
        with self.assertRaisesRegex(exc.RequestMissingFields, 'missing.*id'):
            PredictionService.check_request(X, ['id', 'one', 'two', 'three'])

    def test_check_request_fail_missing_input_columns(self):
        X = pd.DataFrame(
            [[0, 1, 2, 3], [4, 5, 6, 7]],
            columns=['id', 'missing', 'missing', 'three'])
        with self.assertRaisesRegex(exc.RequestMissingFields, 'missing.*one.*two'):
            PredictionService.check_request(X, ['id', 'one', 'two', 'three'])

    def test_check_request_fail_nulls(self):
        X = pd.DataFrame(
            [[0, 1, np.nan, 3], [4, 5, 6, np.nan]],
            columns=['id', 'one', 'two', 'three'])
        with self.assertRaisesRegex(exc.RequestContainsNulls, 'null.*two.*three'):
            PredictionService.check_request(X, ['id', 'one', 'two', 'three'])

    def test_check_request_ignore_nulls_pass(self):
        X = pd.DataFrame(
            [[0, 1, np.nan, 3], [4, 5, 6, np.nan]],
            columns=['id', 'one', 'two', 'three'])
        # no error shoudl be raised
        PredictionService.check_request(X, ['one', 'two', 'three'], True)

    def test_check_request_ignore_nulls_no_check(self):
        # check that the computation counting nulls is never performed
        mock_X = mock.Mock()
        # no error shoudl be raised
        PredictionService.check_request(mock_X, ['one', 'two', 'three'], True)
        mock_X.isnull.assert_not_called()

    @mock.patch('porter.services.PredictionService._default_checks')
    def test_check_request_user_check_fail(self, mock__default_checks):
        X = pd.DataFrame(
            [[0, 1], [4, 0]],
            columns=['id', 'one'])
        class E(Exception): pass
        def additional_checks_fail(X):
            if (X.one == 0).any():
                raise E
        with self.assertRaises(E):
            PredictionService.check_request(X, ['id', 'one', 'two', 'three'], False, additional_checks_fail)


    @mock.patch('flask.request')
    @mock.patch('flask.jsonify')
    @mock.patch('porter.services.BaseService._ids', set())
    def test_get_post_data_batch_prediction(self, mock_flask_jsonify, mock_flask_request):
        mock_model = mock.Mock()
        mock_model.predict.return_value = []

        # Succeed
        mock_flask_request.get_json.return_value = [{'id': None}]
        serve_prediction = PredictionService(
            model=mock_model,
            name=mock.Mock(),
            version=mock.Mock(),
            meta={},
            allow_nulls=mock.Mock(),
            preprocessor=None,
            postprocessor=None,
            batch_prediction=True,
            additional_checks=None
        )
        _ = serve_prediction()

        # Fail
        mock_model = mock.Mock()
        mock_flask_request.get_json.return_value = {'id': None}
        serve_prediction = PredictionService(
            model=mock_model,
            name=mock.Mock(),
            version=mock.Mock(),
            meta={},
            allow_nulls=mock.Mock(),
            preprocessor=None,
            postprocessor=None,
            batch_prediction=True,
            additional_checks=None
        )
        with self.assertRaises(exc.InvalidModelInput):
            _ = serve_prediction()

    @mock.patch('flask.request')
    @mock.patch('flask.jsonify')
    @mock.patch('porter.services.BaseService._ids', set())
    def test_get_post_data_instance_prediction(self, mock_flask_jsonify, mock_flask_request):
        mock_model = mock.Mock()
        mock_model.predict.return_value = [1]

        # Succeed
        mock_flask_request.get_json.return_value = {'id': None}
        serve_prediction = PredictionService(
            model=mock_model,
            name=mock.Mock(),
            version=mock.Mock(),
            meta={},
            allow_nulls=mock.Mock(),
            preprocessor=None,
            postprocessor=None,
            batch_prediction=False,
            additional_checks=None
        )
        _ = serve_prediction()

        # Fail
        mock_model = mock.Mock()
        mock_flask_request.get_json.return_value = [{'id': None}]
        serve_prediction = PredictionService(
            model=mock.Mock(),
            name=mock.Mock(),
            version=mock.Mock(),
            meta={},
            allow_nulls=mock.Mock(),
            preprocessor=None,
            postprocessor=None,
            batch_prediction=False,
            additional_checks=None
        )
        with self.assertRaises(exc.InvalidModelInput):
            _ = serve_prediction()

    @mock.patch('porter.services.PredictionService.reserved_keys', [])
    @mock.patch('porter.services.BaseService._ids', set())
    def test_constructor(self):
        service_config = PredictionService(
            model=None, name='foo', version='bar', meta={'1': '2', '3': 4})

    @mock.patch('porter.services.PredictionService.reserved_keys', ['1', '2'])
    @mock.patch('porter.services.BaseService._ids', set())
    def test_constructor_fail(self):
        with self.assertRaisesRegex(exc.PorterError, 'Could not jsonify meta data'):
            with mock.patch('porter.services.cf.json_encoder', spec={'encode.side_effect': TypeError}) as mock_encoder:
                service_config = PredictionService(
                    model=None, name='foo', version='bar', meta=object())
        with self.assertRaisesRegex(exc.PorterError, '.*keys are reserved for prediction.*'):
            service_config = PredictionService(
                model=None, name='foo', version='bar', meta={'1': '2', '3': 4})
        with self.assertRaisesRegex(exc.PorterError, '.*callable.*'):
            service_config = PredictionService(
                model=None, additional_checks=1)


class TestMiddlewareService(unittest.TestCase):
    @mock.patch('porter.services.MiddlewareService.__init__')
    @mock.patch('porter.services.api.post')
    @mock.patch('porter.services.MiddlewareService.get_post_data')
    @mock.patch('porter.services.porter_responses.make_middleware_response', lambda x: x)
    def test_serve(self, mock_get_post_data, mock_post, mock_init):
        """Test the following
        1. All data from post request is sent to the correct model endpoint.
        2. All corresponding response objects are returned.
        """
        # set up the mocks
        mock_init.return_value = None
        data = enumerate(np.random.randint(0, 100, size=1_000))
        mock_get_post_data.return_value = [{'id': id, 'value': val} for id, val in data]
        def post(url, data):
            return mock.Mock(**{'json.return_value': data['id']})
        mock_post.side_effect = post

        # set up the MiddlewareService instance
        middleware_service = MiddlewareService()
        middleware_service.model_endpoint = 'localhost:5000/'
        middleware_service.max_workers = 2

        # test implementation
        actual = middleware_service.serve()
        expected_calls = [mock.call(middleware_service.model_endpoint, data) for data in data]
        mock_post.post.assert_has_calls(expected_calls)
        # test results
        self.assertEqual(sorted(actual), list(range(1_000)))

    @mock.patch('porter.services.MiddlewareService.__init__')
    @mock.patch('porter.services.api.post')
    @mock.patch('porter.services.MiddlewareService.get_post_data')
    @mock.patch('porter.services.porter_responses.make_middleware_response', lambda x: x)
    def test_serve_with_errors(self, mock_get_post_data, mock_post, mock_init):
        """Test the following
        1. All data from post request is sent to the correct model endpoint.
        2. All corresponding response objects are returned.
        """
        # set up the mocks
        mock_init.return_value = None
        data = enumerate(np.random.randint(0, 100, size=1_000))
        mock_get_post_data.return_value = [{'id': id, 'value': val} for id, val in data]
        def post(url, data):
            if data['id'] % 5:
                return mock.Mock(**{'json.return_value': data['id']})
            return mock.Mock(**{'json.side_effect': ValueError(data['id'])})
        mock_post.side_effect = post

        # set up the MiddlewareService instance
        middleware_service = MiddlewareService()
        middleware_service.model_endpoint = 'localhost:5000/'
        middleware_service.max_workers = 2

        # test implementation
        actual = middleware_service.serve()
        expected_calls = [mock.call(middleware_service.model_endpoint, data) for data in data]
        mock_post.post.assert_has_calls(expected_calls)
        # test results
        actual_ints = sorted(x for x in actual if isinstance(x, int))
        actual_errors = sorted([str(x) for x in actual if isinstance(x, ValueError)])
        expected_ints = [i for i in range(1_000) if i % 5]
        expected_errors = sorted([str(ValueError(i)) for i in range(1_000) if not i % 5])
        self.assertEqual(actual_ints, expected_ints)
        self.assertEqual(actual_errors, expected_errors)

    @mock.patch('porter.services.MiddlewareService.__init__')
    @mock.patch('porter.services.api.request_json')
    def test_get_post_data(self, mock_request_json, mock_init):
        mock_init.return_value = None
        middleware_service = MiddlewareService()
        mock_request_json.return_value = [1, 2]
        actual = middleware_service.get_post_data()
        expected = [1, 2]
        self.assertEqual(actual, expected)

    @mock.patch('porter.services.MiddlewareService.__init__')
    @mock.patch('porter.services.api.request_json')
    def test_get_post_data_fail(self, mock_request_json, mock_init):
        mock_init.return_value = None
        middleware_service = MiddlewareService()
        mock_request_json.return_value = {}
        with self.assertRaisesRegex(exc.InvalidModelInput, 'input must be an array'):
            middleware_service.get_post_data()

    def test_constructor(self):
        middleware_service = MiddlewareService(
            name='a-model',
            version='1.0',
            meta={'foo': 1, 'bar': 'baz'},
            model_endpoint='localhost:5000/a-model/prediction',
            max_workers=20
        )
        expected_id = 'a-model:middleware:1.0'
        expected_endpoint = '/a-model/batchPrediction'
        expected_meta = {'foo': 1, 'bar': 'baz',
                         'model_endpoint': 'localhost:5000/a-model/prediction',
                         'max_workers': 20}
        self.assertEqual(middleware_service.name, 'a-model')
        self.assertEqual(middleware_service.version, '1.0')
        self.assertEqual(middleware_service.id, expected_id)
        self.assertEqual(middleware_service.endpoint, expected_endpoint)
        self.assertEqual(middleware_service.meta, expected_meta)

    @mock.patch('porter.services.MiddlewareService.__init__')
    @mock.patch('porter.services.api.get')
    def test_status_ready(self, mock_get, mock_init):
        mock_init.return_value = None
        mock_get.return_value = mock.Mock(status_code=200)
        middleware_service = MiddlewareService()
        middleware_service.model_endpoint = 'localhost:5000/a-model/prediction'
        self.assertEqual(middleware_service.status, cn.HEALTH_CHECK.RESPONSE.VALUES.STATUS_IS_READY)

    @mock.patch('porter.services.MiddlewareService.__init__')
    @mock.patch('porter.services.api.get')
    def test_status_not_ready1(self, mock_get, mock_init):
        """Model endpoint returns non-200."""
        mock_init.return_value = None
        mock_get.return_value = mock.Mock(status_code=200)
        middleware_service = MiddlewareService()
        middleware_service.model_endpoint = 'localhost:5000/a-model/prediction'
        mock_get.return_value = mock.Mock(status_code=404)
        expected = f'GET {middleware_service.model_endpoint} returned 404'
        self.assertEqual(middleware_service.status, expected)

    @mock.patch('porter.services.MiddlewareService.__init__')
    @mock.patch('porter.services.api.get')
    def test_status_not_ready2(self, mock_get, mock_init):
        """Error is raised when sending HTTP request to model endpoint."""
        mock_init.return_value = None
        mock_get.side_effect = Exception('testing')
        middleware_service = MiddlewareService()
        middleware_service.model_endpoint = 'localhost:5000/a-model/prediction'
        mock_get.return_value = mock.Mock(status_code=404)
        expected = f'cannot communicate with'
        self.assertRegex(middleware_service.status, expected)


class TestModelApp(unittest.TestCase):
    @mock.patch('porter.services.ModelApp._build_app')
    @mock.patch('porter.services.ModelApp.add_service')
    def test_add_services(self, mock_add_service, mock__build_app):
        configs = [object(), object(), object()]
        model_app = ModelApp()
        model_app.add_services(configs[0], configs[1], configs[2])
        expected_calls = [mock.call(obj) for obj in configs]
        mock_add_service.assert_has_calls(expected_calls)

    @mock.patch('porter.services.ModelApp._build_app')
    @mock.patch('porter.services.api.App')
    def test_state(self, mock_App, mock__build_app):
        model_app = ModelApp()
        class service1:
            id = 'service1'
            name = 'foo'
            version = 'bar'
            endpoint = '/an/endpoint'
            meta = {'key1': 'value1', 'key2': 2}
            status = 'ready'
            route_kwargs = {}
        class service2:
            id = 'service2'
            name = 'foobar'
            version = '1'
            endpoint = '/foobar'
            meta = {}
            status = 'ready'
            route_kwargs = {}
        class service3:
            id = 'service3'
            name = 'supa-dupa-model'
            version = '1.0'
            endpoint = '/supa/dupa'
            meta = {'key1': 1}
            status = 'not ready'
            route_kwargs = {}
        model_app.add_service(service1)
        model_app.add_service(service2)
        model_app.add_service(service3)
        actual = model_app.state
        expected = {
            'porter_version': '0.11.0',
            'deployed_on': cn.HEALTH_CHECK.RESPONSE.VALUES.DEPLOYED_ON,
            'services': {
                'service1': {
                    'name': 'foo',
                    'version': 'bar',
                    'endpoint': '/an/endpoint',
                    'meta': {'key1': 'value1', 'key2': 2},
                    'status': 'ready',
                },
                'service2': {
                    'name': 'foobar',
                    'version': '1',
                    'endpoint': '/foobar',
                    'meta': {},
                    'status': 'ready',
                },
                'service3': {
                    'name': 'supa-dupa-model',
                    'version': '1.0',
                    'endpoint': '/supa/dupa',
                    'meta': {'key1': 1},
                    'status': 'not ready',
                },
            }
        }
        self.maxDiff = None
        self.assertDictEqual(actual, expected)

    @mock.patch('porter.services.ModelApp._build_app')
    @mock.patch('porter.services.api.App')
    def test_add_service(self, mock_app, mock__build_app):
        class service1:
            id = 'service1'
            endpoint = '/an/endpoint'
            route_kwargs = {'foo': 1, 'bar': 'baz'}
        class service2:
            id = 'service2'
            endpoint = '/foobar'
            route_kwargs = {'methods': ['GET', 'POST']}
        class service3:
            id = 'service3'
            endpoint = '/supa/dupa'
            route_kwargs = {'methods': ['GET'], 'strict_slashes': True}
        model_app = ModelApp()
        model_app.add_services(service1, service2, service3)
        expected_calls = [
            mock.call('/an/endpoint', foo=1, bar='baz'),
            mock.call()(service1),
            mock.call('/foobar', methods=['GET', 'POST']),
            mock.call()(service2),
            mock.call('/supa/dupa', methods=['GET'], strict_slashes=True),
            mock.call()(service3),
        ]
        model_app.app.route.assert_has_calls(expected_calls)

    @mock.patch('porter.services.ModelApp._build_app')
    @mock.patch('porter.services.api.App')
    def test_add_service_fail(self, mock_app, mock__build_app):
        class service1:
            id = 'service1'
            endpoint = '/an/endpoint'
            route_kwargs = {}
        class service2:
            id = 'service1'
            endpoint = '/foobar'
            route_kwargs = {}
        model_app = ModelApp()
        model_app.add_service(service1)
        with self.assertRaisesRegex(exc.PorterError, 'service has already been added'):
            model_app.add_service(service2)


class TestBaseService(unittest.TestCase):
    @mock.patch('porter.services.BaseService.define_endpoint')
    def test_constructor(self, mock_define_endpoint):
        # test ABC
        with self.assertRaisesRegex(TypeError, 'abstract methods'):
            class SC(BaseService):
                def define_endpoint(self):
                    return '/an/endpoint'
            SC()

        class SC(BaseService):
            def define_endpoint(self):
                return '/an/endpoint'
            def serve(self): pass
            def status(self): pass


        with self.assertRaisesRegex(exc.PorterError, 'Could not jsonify meta data'):
            with mock.patch('porter.services.cf.json_encoder', spec={'encode.side_effect': TypeError}) as mock_encoder:
                SC(name='foo', version='bar', meta=object())
        service_config = SC(name='foo', version='bar', meta=None)
        self.assertEqual(service_config.endpoint, '/an/endpoint')
        # make sure this gets set -- shouldn't raise AttributeError
        service_config.id
        # make sure that creating a config with same name and version raises
        # error
        with self.assertRaisesRegex(exc.PorterError, '.*likely means that you tried to instantiate a service.*'):
            service_config = SC(name='foo', version='bar', meta=None)


if __name__ == '__main__':
    unittest.main()
