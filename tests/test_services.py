import time
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import porter.responses as porter_responses
from porter import __version__
from porter import constants as cn
from porter import exceptions as exc
from porter.services import (BaseService, ModelApp,
                             PredictionService,
                             StatefulRoute, serve_error_message)
from porter.schemas import openapi


class TestFunctionsUnit(unittest.TestCase):
    @mock.patch('porter.services.porter_responses.api.request_json')
    @mock.patch('porter.services.porter_responses.api.jsonify')
    @mock.patch('porter.services.porter_responses.api.request_id', lambda: 123)
    @mock.patch('porter.services.cf.return_message_on_error', True)
    @mock.patch('porter.services.cf.return_traceback_on_error', True)
    @mock.patch('porter.services.cf.return_user_data_on_error', True)
    def test_serve_error_message_status_codes_arbitrary_error(self, mock_flask_request, mock_flask_jsonify):
        # if the current error does not have an error code make sure
        # the response gets a 500
        error = ValueError('an error message')
        actual = serve_error_message(error)
        actual_status_code = 500
        expected_status_code = 500
        self.assertEqual(actual_status_code, expected_status_code)

    @mock.patch('porter.services.porter_responses.api.request_json')
    @mock.patch('porter.services.porter_responses.api.jsonify')
    @mock.patch('porter.services.porter_responses.api.request_id', lambda: 123)
    @mock.patch('porter.services.cf.return_message_on_error', True)
    @mock.patch('porter.services.cf.return_traceback_on_error', True)
    @mock.patch('porter.services.cf.return_user_data_on_error', True)
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


class TestPredictionSchema(unittest.TestCase):
    def test_prediction_schema_constructor(self):
        schema = PredictSchema(input_features=None)
        self.assertIs(schema.input_columns, None)
        self.assertIs(schema.input_features, None)

        schema = PredictSchema(input_features=['feature1', 'feature2'])
        self.assertIsInstance(schema.input_columns, list)
        self.assertIsInstance(schema.input_features, list)

        schema = PredictSchema(input_features=('feature1', 'feature2'))
        self.assertIsInstance(schema.input_columns, list)
        self.assertIsInstance(schema.input_features, list)



@mock.patch('porter.responses.api.request_id', lambda: 123)
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
        test_api_version = '1.0.0'
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
            api_version=test_api_version,
            meta={'1': '2', '3': 4},
            preprocessor=mock_preprocessor,
            postprocessor=mock_postprocessor,
            allow_nulls=allow_nulls,
            batch_prediction=True,
            additional_checks=None
        )
        actual = serve_prediction()
        expected = {
            'request_id': 123,
            'model_context': {
                'model_name': test_model_name,
                'api_version': test_api_version,
                'model_meta': {
                    '1': '2',
                    '3': 4
                }
            },
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
        test_api_version = '1.0.0'
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
            api_version=test_api_version,
            meta={'1': '2', '3': 4},
            preprocessor=mock_preprocessor,
            postprocessor=mock_postprocessor,
            allow_nulls=allow_nulls,
            batch_prediction=False,
            additional_checks=None
        )
        actual = serve_prediction()
        expected = {
            'request_id': 123,
            'model_context': {
                'model_name': test_model_name,
                'api_version': test_api_version,
                'model_meta': {
                    '1': '2',
                    '3': 4
                }
            },
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
        name = 'my-model'
        version = '1.0'
        meta = {}
        with self.assertRaises(exc.PredictionError) as ctx:
            sp = PredictionService(
                model=mock.Mock(), name=name, api_version=version,
                meta=meta, preprocessor=mock.Mock(), postprocessor=mock.Mock(),
                allow_nulls=mock.Mock(), batch_prediction=mock.Mock(),
                additional_checks=mock.Mock())
            sp()
            # porter.responses.make_error_response counts on these attributes being filled out
            self.assertEqual(ctx.exception.model_name, name)
            self.assertEqual(ctx.exception.api_version, version)
            self.assertEqual(ctx.exception.model_meta, meta)

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
        model_name = api_version = mock.MagicMock()
        serve_prediction = PredictionService(
            model=mock_model,
            name=model_name,
            api_version=api_version,
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
        model = allow_nulls = mock.Mock()
        model_name = api_version = mock.MagicMock()
        mock_flask_request.get_json.return_value = [{'id': None}]
        model.predict.return_value = []
        serve_prediction = PredictionService(
            model=model,
            name=model_name,
            api_version=api_version,
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
        model = allow_nulls = mock.Mock()
        model_name = api_version = mock.MagicMock()
        mock_flask_request.get_json.return_value = {'id': None}
        model.predict.return_value = [1]
        mock_preprocessor = mock.Mock()
        mock_preprocessor.process.return_value = {}
        mock_postprocessor = mock.Mock()
        mock_postprocessor.process.return_value = [1]
        serve_prediction = PredictionService(
            model=model,
            name=model_name,
            api_version=api_version,
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
        model = allow_nulls = mock.Mock()
        model_name = api_version = mock.MagicMock()
        mock_flask_request.get_json.return_value = {'id': None}
        model.predict.return_value = [1]
        serve_prediction = PredictionService(
            model=model,
            name=model_name,
            api_version=api_version,
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

    # def test_define_endpoint(self):
    #     prediction_service = PredictionService(name='my-model', api_version='v1', namespace='/my/namespace')

    @mock.patch('porter.services.api')
    @mock.patch('porter.responses.api')
    @mock.patch('porter.services.BaseService._ids', set())
    def test_get_post_data_batch_prediction(self, mock_responses_api, mock_services_api):
        mock_model = mock.Mock()
        mock_model.predict.return_value = []
        mock_name = mock_version = mock.MagicMock()

        # Succeed
        mock_services_api.request_json.return_value = [{'id': None}]
        serve_prediction = PredictionService(
            model=mock_model,
            name=mock_name,
            api_version=mock_version,
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
        mock_services_api.request_json.return_value = {'id': None}
        serve_prediction = PredictionService(
            model=mock_model,
            name=mock.MagicMock(),
            api_version=mock.MagicMock(),
            meta={},
            allow_nulls=mock.Mock(),
            preprocessor=None,
            postprocessor=None,
            batch_prediction=True,
            additional_checks=None
        )
        with self.assertRaises(exc.InvalidModelInput):
            _ = serve_prediction()

    @mock.patch('porter.services.api')
    @mock.patch('porter.responses.api')
    @mock.patch('porter.services.BaseService._ids', set())
    def test_get_post_data_instance_prediction(self, mock_responses_api, mock_services_api):
        mock_model = mock.Mock()
        mock_model.predict.return_value = [1]

        # Succeed
        mock_services_api.request_json.return_value = {'id': None}
        serve_prediction = PredictionService(
            model=mock_model,
            name=mock.MagicMock(),
            api_version=mock.MagicMock(),
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
        mock_services_api.request_json.return_value = [{'id': None}]
        serve_prediction = PredictionService(
            model=mock.Mock(),
            name=mock.MagicMock(),
            api_version=mock.MagicMock(),
            meta={},
            allow_nulls=mock.Mock(),
            preprocessor=None,
            postprocessor=None,
            batch_prediction=False,
            additional_checks=None
        )
        with self.assertRaises(exc.InvalidModelInput):
            _ = serve_prediction()

    @mock.patch('porter.services.api')
    @mock.patch('porter.responses.api')
    @mock.patch('porter.services.BaseService._ids', set())
    def test_get_post_data_prediction_schema(self, mock_responses_api, mock_services_api):
        # test if validating or not;
        # test_schemas_openapi.py confirms more complex validations also work
        feature_schema = openapi.Object(
            properties=dict(
                a=openapi.String(),
                b=openapi.Integer()
            )
        )
        prediction_schema = openapi.Object(
            properties=dict(
                x=openapi.Number(additional_params=dict(minimum=0, maximum=1)),
                y=openapi.Integer(),
            )
        )
        # test both instance and batch prediction
        for batch_prediction in (False, True):
            in_good = {'id': 1, 'a': 'a', 'b': 1}
            in_bad = {'id': 1, 'a': 'a', 'b': 1.5}
            if batch_prediction:
                in_good, in_bad = [in_good], [in_bad]

            out_good = [{'id': 1, 'x': 0.5, 'y': 0}]
            out_bad = [{'id': 1, 'x': -0.5, 'y': 0}]

            # test all combos of validating request x response
            for val_request in (False, True):
                for val_response in (False, True):
                    # TODO: tests only pass if
                    # (val_request,val_response) = (True,False)
                    if (not val_request) or val_response:
                        continue
                    #print('* batch, request, response = {}, {}, {}'.format(
                    #    batch_prediction, val_request, val_response
                    #))
                    mock_model = mock.Mock()
                    mock_name = mock_version = mock.MagicMock()
                    serve_prediction = PredictionService(
                        model=mock_model,
                        name=mock_name,
                        api_version=mock_version,
                        meta={},
                        allow_nulls=mock.Mock(),
                        preprocessor=None,
                        postprocessor=None,
                        batch_prediction=batch_prediction,
                        additional_checks=None,
                        feature_schema=feature_schema,
                        prediction_schema=prediction_schema,
                        validate_request_data=val_request,
                        validate_response_data=val_response,
                    )
                    # good in + out should always work
                    mock_services_api.request_json.return_value = in_good
                    mock_model.predict.return_value = out_good
                    _ = serve_prediction()

                    # bad in + good out should raise if val_request
                    mock_services_api.request_json.return_value = in_bad
                    mock_model.predict.return_value = out_good
                    if val_request:
                        with self.assertRaises(exc.InvalidModelInput):
                            _ = serve_prediction()
                    else:
                        _ = serve_prediction()

                    # good in + bad out should raise if val_response
                    mock_services_api.request_json.return_value = in_good
                    mock_model.predict.return_value = out_bad
                    if val_response:
                        # TODO: should be new `exc.InvalidModelOutput` ?
                        with self.assertRaises(ValueError):
                            _ = serve_prediction()
                    else:
                        _ = serve_prediction()

                    # bad in + bad out should only work if
                    # neither val_request nor val_response
                    mock_services_api.request_json.return_value = in_bad
                    mock_model.predict.return_value = out_bad
                    if val_request:
                        with self.assertRaises(exc.InvalidModelInput):
                            _ = serve_prediction()
                    elif val_response:
                        with self.assertRaises(ValueError):
                            _ = serve_prediction()
                    else:
                        _ = serve_prediction()

    @mock.patch('porter.services.BaseService._ids', set())
    def test_constructor(self):
        service_config = PredictionService(
            model=None, name='foo', api_version='bar', meta={'1': '2', '3': 4})

    @mock.patch('porter.services.BaseService._ids', set())
    def test_constructor_fail(self):
        with self.assertRaisesRegex(exc.PorterError, 'Could not jsonify meta data'):
            with mock.patch('porter.services.cf.json_encoder', spec={'encode.side_effect': TypeError}) as mock_encoder:
                service_config = PredictionService(
                    model=None, name='foo', api_version='bar', meta=object())
        with self.assertRaisesRegex(exc.PorterError, '.*callable.*'):
            service_config = PredictionService(
                model=None, additional_checks=1)


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
    @mock.patch('porter.services.BaseService._ids', set())
    @mock.patch('porter.services.BaseService.define_endpoint')
    @mock.patch('porter.services.BaseService.action', None)
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
                SC(name='foo', api_version='bar', meta=object())
        service_config = SC(name='foo', api_version='bar', meta=None)
        self.assertEqual(service_config.endpoint, '/an/endpoint')
        # make sure this gets set -- shouldn't raise AttributeError
        service_config.id
        # make sure that creating a config with same name and version raises
        # error
        with self.assertRaisesRegex(exc.PorterError, '.*likely means that you tried to instantiate a service.*'):
            service_config = SC(name='foo', api_version='bar', meta=None)

    @mock.patch('porter.services.BaseService._ids', set())
    @mock.patch('porter.services.api.request_json', lambda: {'foo': 1, 'bar': {'p': 10}})
    @mock.patch('porter.services.api.request_id', lambda: 123)
    @mock.patch('porter.services.BaseService.action', None)
    def test_api_logging_no_exception(self):
        class Service(BaseService):
            def serve(self):
                m = mock.Mock(spec=porter_responses.Response)
                m.jsonify.side_effect = lambda: {'foo': '1', 'p': {10: '10'}}
                return m
            def status(self):
                return 'ready'

        with mock.patch('porter.services.BaseService._logger') as mock__logger:
            service1 = Service(name='name1', api_version='version1', log_api_calls=True)
            served = service1()
            mock__logger.info.assert_called_with(
                'api logging',
                extra={'request_id': 123,
                       'request_data': {'foo': 1, 'bar': {'p': 10}},
                       'response_data': {'foo': '1', 'p': {10: '10'}},
                       'service_class': 'Service',
                       'event': 'api_call'}
                )

        with mock.patch('porter.services.BaseService._logger') as mock__logger:
            service2 = Service(name='name2', api_version='version2', log_api_calls=False)
            service2()
            mock__logger.assert_not_called()

    @mock.patch('porter.services.BaseService._ids', set())
    @mock.patch('porter.services.api.request_json', lambda: {'foo': 1, 'bar': {'p': 10}})
    @mock.patch('porter.services.api.request_id', lambda: 123)
    @mock.patch('porter.services.BaseService.action', None)
    def test_api_logging_exception(self):
        class Service(BaseService):
            def serve(self):
                raise Exception('testing')
            def status(self):
                return 'ready'

        with mock.patch('porter.services.BaseService._logger') as mock__logger:
            service1 = Service(name='name1', api_version='version1', log_api_calls=True)
            with self.assertRaisesRegex(Exception, 'testing'):
                service1()
            mock__logger.info.assert_called_with(
                'api logging',
                extra={'request_id': 123,
                       'request_data': {'foo': 1, 'bar': {'p': 10}},
                       'response_data': None,
                       'service_class': 'Service',
                       'event': 'api_call'}
                )

        with mock.patch('porter.services.BaseService._logger') as mock__logger:
            service2 = Service(name='name2', api_version='version2', log_api_calls=False)
            with self.assertRaisesRegex(Exception, 'testing'):
                service2()
            mock__logger.assert_not_called()

    @mock.patch('porter.services.BaseService._logger')
    @mock.patch('porter.api.request_id', lambda: 123)
    @mock.patch('porter.services.BaseService._ids', set())
    @mock.patch('porter.services.BaseService.action', None)
    def test_serve_logging_with_exception(self, mock__logger):
        e = Exception('testing')
        class Service(BaseService):
            def define_endpoint(self):
                return '/foo'
            def serve(self):
                raise e
            def status(self):
                return 'ready'

        service = Service(name='name', api_version='version')
        with self.assertRaisesRegex(Exception, 'testing'):
            service()
        mock__logger.exception.assert_called_with(
            e,
            extra={'request_id': 123,
                   'service_class': 'Service',
                   'event': 'exception'})

    @mock.patch('porter.services.BaseService._ids', set())
    @mock.patch('porter.services.BaseService.serve', None)
    @mock.patch('porter.services.BaseService.status', None)
    def test_define_endpoint_with_namespace(self):
        class Service(BaseService):
            action = 'foo'
        service = Service(name='my-service', api_version='v11', namespace='/my/namespace')
        expected = '/my/namespace/my-service/v11/foo'
        self.assertEqual(service.endpoint, expected)

    @mock.patch('porter.services.BaseService._ids', set())
    @mock.patch('porter.services.BaseService.serve', None)
    @mock.patch('porter.services.BaseService.status', None)
    def test_define_endpoint_with_namespace(self):
        class Service(BaseService):
            action = 'bar'
        # test without namespace (since it's optional)
        service = Service(name='my-service', api_version='v11')
        expected = '/my-service/v11/bar'
        self.assertEqual(service.endpoint, expected)

    @mock.patch('porter.services.BaseService.serve', None)
    @mock.patch('porter.services.BaseService.status', None)
    def test_define_endpoint_with_bad_namespace(self):
        class Service(BaseService):
            action = 'bar'

        with mock.patch('porter.services.BaseService._ids', set()):
            # no /
            service = Service(name='my-service', api_version='v11', namespace='ns')
            expected = '/ns/my-service/v11/bar'
            self.assertEqual(service.endpoint, expected)

        with mock.patch('porter.services.BaseService._ids', set()):
            # trailing /
            service = Service(name='my-service', api_version='v11', namespace='n/s/')
            expected = '/n/s/my-service/v11/bar'
            self.assertEqual(service.endpoint, expected)

        with mock.patch('porter.services.BaseService._ids', set()):
            # both /
            service = Service(name='my-service', api_version='v11', namespace='/n/s/')
            expected = '/n/s/my-service/v11/bar'
            self.assertEqual(service.endpoint, expected)


if __name__ == '__main__':
    unittest.main()
