import json
import unittest
from unittest import mock

import numpy as np
import pandas as pd
from porter.exceptions import PredictionError
from porter.services import (BaseServiceConfig, ModelApp,
                             PredictionServiceConfig, ServePrediction,
                             StatefulRoute, serve_error_message)


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


class TestServePrediction(unittest.TestCase):
    @mock.patch('flask.request')
    @mock.patch('porter.responses.flask')
    def test_serve_success_batch(self, mock_responses_flask, mock_flask_request):
        mock_flask_request.get_json.return_value = [
            {'id': 1, 'feature1': 10, 'feature2': 0},
            {'id': 2, 'feature1': 11, 'feature2': 1},
            {'id': 3, 'feature1': 12, 'feature2': 2},
            {'id': 4, 'feature1': 13, 'feature2': 3},
            {'id': 5, 'feature1': 14, 'feature2': 3},
        ]
        mock_responses_flask.jsonify = lambda payload:payload
        mock_model = mock.Mock()
        test_model_name = 'model'
        test_model_version = '1.0.0'
        mock_preprocessor = mock.Mock()
        mock_postprocessor = mock.Mock()
        schema = mock.Mock(input_features=None, input_columns=None)
        allow_nulls = False

        feature_values = {str(x): x for x in range(5)}
        mock_model.predict = lambda X: X['feature1'] + X['feature2'].map(feature_values) + X['feature3']
        def preprocess(X):
            X['feature2'] = X.feature2.astype(str)
            X['feature3'] = range(len(X))
            return X
        mock_preprocessor.process = preprocess
        def postprocess(X):
            return X * 2
        mock_postprocessor.process = postprocess
        serve_prediction = ServePrediction(
            model=mock_model,
            model_name=test_model_name,
            model_version=test_model_version,
            model_meta={1: '2', '3': 4},
            preprocessor=mock_preprocessor,
            postprocessor=mock_postprocessor,
            schema=schema,
            allow_nulls=allow_nulls,
            batch_prediction=True,
        )
        actual = serve_prediction()
        expected = {
            'model_name': test_model_name,
            'model_version': test_model_version,
            1: '2',
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

    @mock.patch('flask.request')
    @mock.patch('porter.responses.flask')
    def test_serve_success_single(self, mock_responses_flask, mock_flask_request):
        mock_flask_request.get_json.return_value = {'id': 1, 'feature1': 10, 'feature2': 0}
        mock_responses_flask.jsonify = lambda payload:payload
        mock_model = mock.Mock()
        test_model_name = 'model'
        test_model_version = '1.0.0'
        mock_preprocessor = mock.Mock()
        mock_postprocessor = mock.Mock()
        schema = mock.Mock(input_features=None, input_columns=None)
        allow_nulls = False

        feature_values = {str(x): x for x in range(5)}
        mock_model.predict = lambda X: X['feature1'] + X['feature2'].map(feature_values) + X['feature3']
        def preprocess(X):
            X['feature2'] = X.feature2.astype(str)
            X['feature3'] = range(len(X))
            return X
        mock_preprocessor.process = preprocess
        def postprocess(X):
            return X * 2
        mock_postprocessor.process = postprocess
        serve_prediction = ServePrediction(
            model=mock_model,
            model_name=test_model_name,
            model_version=test_model_version,
            model_meta={1: '2', '3': 4},
            preprocessor=mock_preprocessor,
            postprocessor=mock_postprocessor,
            schema=schema,
            allow_nulls=allow_nulls,
            batch_prediction=False,
        )
        actual = serve_prediction()
        expected = {
            'model_name': test_model_name,
            'model_version': test_model_version,
            1: '2',
            '3': 4,
            'predictions': {'id': 1, 'prediction': 20}
        }
        self.assertEqual(actual, expected)

    @mock.patch('porter.services.ServePrediction._predict')
    def test_serve_fail(self, mock__predict):
        mock__predict.side_effect = Exception
        with self.assertRaises(PredictionError):
            sp = ServePrediction(
                model=mock.Mock(), model_name=mock.Mock(), model_version=mock.Mock(),
                model_meta=mock.Mock(), preprocessor=mock.Mock(), postprocessor=mock.Mock(),
                schema=mock.Mock(), allow_nulls=mock.Mock(), batch_prediction=mock.Mock())
            sp()

    @mock.patch('flask.request')
    @mock.patch('flask.jsonify')
    def test_serve_with_processing_batch(self, mock_flask_jsonify, mock_flask_request):
        model = model_name = model_version = allow_nulls = mock.Mock()
        mock_flask_request.get_json.return_value = [{'id': None}]
        model.predict.return_value = []
        mock_preprocessor = mock.Mock()
        mock_preprocessor.process.return_value = {}
        mock_postprocessor = mock.Mock()
        mock_postprocessor.process.return_value = []
        mock_schema = mock.Mock(input_features=None, input_columns=None)
        serve_prediction = ServePrediction(
            model=model,
            model_name=model_name,
            model_version=model_version,
            model_meta={},
            schema=mock_schema,
            allow_nulls=allow_nulls,
            preprocessor=mock_preprocessor,
            postprocessor=mock_postprocessor,
            batch_prediction=True,
        )
        _ = serve_prediction()
        mock_preprocessor.process.assert_called()
        mock_postprocessor.process.assert_called()

    @mock.patch('flask.request')
    @mock.patch('flask.jsonify')
    def test_serve_no_processing_batch(self, mock_flask_jsonify, mock_flask_request):
        # make sure it doesn't break when processors are None
        model = model_name = model_version = allow_nulls = mock.Mock()
        mock_schema = mock.Mock(input_features=None, input_columns=None)
        mock_flask_request.get_json.return_value = [{'id': None}]
        model.predict.return_value = []
        serve_prediction = ServePrediction(
            model=model,
            model_name=model_name,
            model_version=model_version,
            model_meta={},
            schema=mock_schema,
            allow_nulls=allow_nulls,
            preprocessor=None,
            postprocessor=None,
            batch_prediction=True
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
        mock_schema = mock.Mock(input_features=None, input_columns=None)
        serve_prediction = ServePrediction(
            model=model,
            model_name=model_name,
            model_version=model_version,
            model_meta={},
            schema=mock_schema,
            allow_nulls=allow_nulls,
            preprocessor=mock_preprocessor,
            postprocessor=mock_postprocessor,
            batch_prediction=False,
        )
        _ = serve_prediction()
        mock_preprocessor.process.assert_called()
        mock_postprocessor.process.assert_called()

    @mock.patch('flask.request')
    @mock.patch('flask.jsonify')
    def test_serve_no_processing_single(self, mock_flask_jsonify, mock_flask_request):
        # make sure it doesn't break when processors are None
        model = model_name = model_version = allow_nulls = mock.Mock()
        mock_schema = mock.Mock(input_features=None, input_columns=None)
        mock_flask_request.get_json.return_value = {'id': None}
        model.predict.return_value = [1]
        serve_prediction = ServePrediction(
            model=model,
            model_name=model_name,
            model_version=model_version,
            model_meta={},
            schema=mock_schema,
            allow_nulls=allow_nulls,
            preprocessor=None,
            postprocessor=None,
            batch_prediction=False
        )
        _ = serve_prediction()

    def test_check_request_pass(self):
        # no error should be raised
        X = pd.DataFrame(
            [[0, 1, 2, 3], [4, 5, 6, 7]],
            columns=['id', 'one', 'two', 'three'])
        ServePrediction.check_request(X, ['id', 'one', 'two', 'three'])

    def test_check_request_fail_missing_id(self):
        X = pd.DataFrame(
            [[0, 1, 2, 3], [4, 5, 6, 7]],
            columns=['missing', 'one', 'two', 'three'])
        with self.assertRaises(ValueError):
            ServePrediction.check_request(X, ['id', 'one', 'two', 'three'])

    def test_check_request_fail_missing_id_column(self):
        X = pd.DataFrame(
            [[0, 1, 2, 3], [4, 5, 6, 7]],
            columns=['missing', 'one', 'two', 'three'])
        with self.assertRaisesRegexp(ValueError, 'missing.*id'):
            ServePrediction.check_request(X, ['id', 'one', 'two', 'three'])

    def test_check_request_fail_missing_input_columns(self):
        X = pd.DataFrame(
            [[0, 1, 2, 3], [4, 5, 6, 7]],
            columns=['id', 'missing', 'missing', 'three'])
        with self.assertRaisesRegexp(ValueError, 'missing.*one.*two'):
            ServePrediction.check_request(X, ['id', 'one', 'two', 'three'])

    def test_check_request_fail_nulls(self):
        X = pd.DataFrame(
            [[0, 1, np.nan, 3], [4, 5, 6, np.nan]],
            columns=['id', 'one', 'two', 'three'])
        with self.assertRaisesRegexp(ValueError, 'null.*two.*three'):
            ServePrediction.check_request(X, ['id', 'one', 'two', 'three'])

    def test_check_request_ignore_nulls_pass(self):
        X = pd.DataFrame(
            [[0, 1, np.nan, 3], [4, 5, 6, np.nan]],
            columns=['id', 'one', 'two', 'three'])
        # no error shoudl be raised
        ServePrediction.check_request(X, ['one', 'two', 'three'], True)

    def test_check_request_ignore_nulls_no_check(self):
        # check that the computation counting nulls is never performed
        mock_X = mock.Mock()
        # no error shoudl be raised
        ServePrediction.check_request(mock_X, ['one', 'two', 'three'], True)
        mock_X.isnull.assert_not_called()

    @mock.patch('flask.request')
    @mock.patch('flask.jsonify')
    def test_get_post_data_batch_prediction(self, mock_flask_jsonify, mock_flask_request):
        model = model_name = model_version = allow_nulls = mock.Mock()
        mock_schema = mock.Mock(input_features=None, input_columns=None)
        model.predict.return_value = []

        # Succeed
        mock_flask_request.get_json.return_value = [{'id': None}]
        serve_prediction = ServePrediction(
            model=model,
            model_name=model_name,
            model_version=model_version,
            model_meta={},
            schema=mock_schema,
            allow_nulls=allow_nulls,
            preprocessor=None,
            postprocessor=None,
            batch_prediction=True
        )
        _ = serve_prediction()

        # Fail
        mock_flask_request.get_json.return_value = {'id': None}
        serve_prediction = ServePrediction(
            model=model,
            model_name=model_name,
            model_version=model_version,
            model_meta={},
            schema=mock_schema,
            allow_nulls=allow_nulls,
            preprocessor=None,
            postprocessor=None,
            batch_prediction=True
        )
        with self.assertRaises(PredictionError):
            _ = serve_prediction()

    @mock.patch('flask.request')
    @mock.patch('flask.jsonify')
    def test_get_post_data_instance_prediction(self, mock_flask_jsonify, mock_flask_request):
        model = model_name = model_version = allow_nulls = mock.Mock()
        mock_schema = mock.Mock(input_features=None, input_columns=None)
        model.predict.return_value = [1]

        # Succeed
        mock_flask_request.get_json.return_value = {'id': None}
        serve_prediction = ServePrediction(
            model=model,
            model_name=model_name,
            model_version=model_version,
            model_meta={},
            schema=mock_schema,
            allow_nulls=allow_nulls,
            preprocessor=None,
            postprocessor=None,
            batch_prediction=False
        )
        _ = serve_prediction()

        # Fail
        mock_flask_request.get_json.return_value = [{'id': None}]
        serve_prediction = ServePrediction(
            model=model,
            model_name=model_name,
            model_version=model_version,
            model_meta={},
            schema=mock_schema,
            allow_nulls=allow_nulls,
            preprocessor=None,
            postprocessor=None,
            batch_prediction=False
        )
        with self.assertRaises(PredictionError):
            _ = serve_prediction()


class TestModelApp(unittest.TestCase):
    @mock.patch('porter.services.ModelApp._build_app')
    @mock.patch('porter.services.ModelApp.add_service')
    def test_add_services(self, mock_add_service, mock__build_app):
        configs = [object(), object(), object()]
        model_app = ModelApp()
        model_app.add_services(configs[0], configs[1], configs[2])
        expected_calls = [mock.call(obj) for obj in configs]
        mock_add_service.assert_has_calls(expected_calls)


class TestBaseServiceConfig(unittest.TestCase):
    @mock.patch('porter.services.BaseServiceConfig.define_endpoint')
    def test_constructor(self, mock_define_endpoint):
        mock_define_endpoint.return_value = '/an-endpoint'
        with self.assertRaisesRegexp(ValueError, 'Could not jsonify meta data'):
            BaseServiceConfig(name='foo', version='bar', meta=object())
        service_config = BaseServiceConfig(name='foo', version='bar', meta=None)
        self.assertEqual(service_config.endpoint, '/an-endpoint')
        # make sure this gets set -- shouldn't raise AttributeError
        service_config.id
        # make sure that creating a config with same name and version raises
        # error
        with self.assertRaisesRegexp(ValueError, '.*likely means that you tried to instantiate a service.*'):
            service_config = BaseServiceConfig(name='foo', version='bar', meta=None)


class TestPredictionServiceConfig(unittest.TestCase):
    @mock.patch('porter.services.PredictionServiceConfig.reserved_keys', [])
    @mock.patch('porter.services.BaseServiceConfig._ids', set())
    def test_constructor(self):
        service_config = PredictionServiceConfig(
            model=None, name='foo', version='bar', meta={1: '2', '3': 4})

    @mock.patch('porter.services.PredictionServiceConfig.reserved_keys', [1, 2])
    @mock.patch('porter.services.BaseServiceConfig._ids', set())
    def test_constructor_fail(self):
        with self.assertRaisesRegexp(ValueError, 'Could not jsonify meta data'):
            service_config = PredictionServiceConfig(
                model=None, name='foo', version='bar', meta=object())
        with self.assertRaisesRegexp(ValueError, '.*keys are reserved for prediction.*'):
            service_config = PredictionServiceConfig(
                model=None, name='foo', version='bar', meta={1: '2', '3': 4})


if __name__ == '__main__':
    unittest.main()
