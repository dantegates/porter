import json
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from porter.constants import KEYS
from porter.services import (ModelApp, NumpyEncoder, ServePrediction,
                             serve_error_message, StatefulRoute)


class TestFuntionsUnit(unittest.TestCase):
    @mock.patch('flask.jsonify')
    def test_serve_error_message_status_codes(self, mock_flask_jsonify):
        error = ValueError('an error message')
        actual = serve_error_message(error)
        actual_status_code = 500
        expected_status_code = 500
        self.assertEqual(actual_status_code, expected_status_code)

    @mock.patch('flask.jsonify')
    def test_serve_error_message_status_codes(self, mock_flask_jsonify):
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
    def test_serve_success(self, mock_responses_flask, mock_flask_request):
        mock_flask_request.get_json.return_value = [
            {KEYS.PREDICTION.ID: 1, 'feature1': 10, 'feature2': 0},
            {KEYS.PREDICTION.ID: 2, 'feature1': 11, 'feature2': 1},
            {KEYS.PREDICTION.ID: 3, 'feature1': 12, 'feature2': 2},
            {KEYS.PREDICTION.ID: 4, 'feature1': 13, 'feature2': 3},
            {KEYS.PREDICTION.ID: 5, 'feature1': 14, 'feature2': 3},
        ]
        mock_responses_flask.jsonify = lambda payload:payload
        mock_model = mock.Mock()
        test_model_id = 'model.id'
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
            model_id=test_model_id,
            preprocessor=mock_preprocessor,
            postprocessor=mock_postprocessor,
            schema=schema,
            allow_nulls=allow_nulls
        )
        actual = serve_prediction()
        expected = {
            'model_id': test_model_id,
            'predictions': [
                {'id': 1, 'prediction': 20},
                {'id': 2, 'prediction': 26},
                {'id': 3, 'prediction': 32},
                {'id': 4, 'prediction': 38},
                {'id': 5, 'prediction': 42},
            ]
        }
        self.assertEqual(actual[KEYS.PREDICTION.MODEL_ID], expected[KEYS.PREDICTION.MODEL_ID])
        self.assertEqual(sorted(actual[KEYS.PREDICTION.PREDICTIONS], key=lambda x: x[KEYS.PREDICTION.ID]),
                         sorted(expected[KEYS.PREDICTION.PREDICTIONS], key=lambda x: x[KEYS.PREDICTION.ID]))

    @mock.patch('flask.request')
    @mock.patch('flask.jsonify')
    def test_serve_with_processing(self, mock_flask_jsonify, mock_flask_request):
        model = model_id = allow_nulls = mock.Mock()
        mock_flask_request.get_json.return_value = {KEYS.PREDICTION.ID: []}
        model.predict.return_value = []
        mock_preprocessor = mock.Mock()
        mock_preprocessor.process.return_value = {}
        mock_postprocessor = mock.Mock()
        mock_postprocessor.process.return_value = []
        mock_schema = mock.Mock(input_features=None, input_columns=None)
        serve_prediction = ServePrediction(
            model=model,
            model_id=model_id,
            schema=mock_schema,
            allow_nulls=allow_nulls,
            preprocessor=mock_preprocessor,
            postprocessor=mock_postprocessor
        )
        _ = serve_prediction()
        mock_preprocessor.process.assert_called()
        mock_postprocessor.process.assert_called()

    @mock.patch('flask.request')
    @mock.patch('flask.jsonify')
    def test_serve_no_processing(self, mock_flask_jsonify, mock_flask_request):
        # make sure it doesn't break when processors are None
        model = model_id = allow_nulls = mock.Mock()
        mock_schema = mock.Mock(input_features=None, input_columns=None)
        mock_flask_request.get_json.return_value = {KEYS.PREDICTION.ID: []}
        model.predict.return_value = []
        serve_prediction = ServePrediction(
            model=model,
            model_id=model_id,
            schema=mock_schema,
            allow_nulls=allow_nulls,
            preprocessor=None,
            postprocessor=None
        )
        _ = serve_prediction()


    def test_check_request_pass(self):
        # no error should be raised
        X = pd.DataFrame(
            [[0, 1, 2, 3], [4, 5, 6, 7]],
            columns=[KEYS.PREDICTION.ID, 'one', 'two', 'three'])
        ServePrediction.check_request(X, [KEYS.PREDICTION.ID, 'one', 'two', 'three'])

    def test_check_request_fail_missing_id(self):
        X = pd.DataFrame(
            [[0, 1, 2, 3], [4, 5, 6, 7]],
            columns=['missing', 'one', 'two', 'three'])
        with self.assertRaises(ValueError):
            ServePrediction.check_request(X, [KEYS.PREDICTION.ID, 'one', 'two', 'three'])

    def test_check_request_fail_missing_id_column(self):
        X = pd.DataFrame(
            [[0, 1, 2, 3], [4, 5, 6, 7]],
            columns=['missing', 'one', 'two', 'three'])
        with self.assertRaisesRegexp(ValueError, 'missing.*id'):
            ServePrediction.check_request(X, [KEYS.PREDICTION.ID, 'one', 'two', 'three'])

    def test_check_request_fail_missing_input_columns(self):
        X = pd.DataFrame(
            [[0, 1, 2, 3], [4, 5, 6, 7]],
            columns=[KEYS.PREDICTION.ID, 'missing', 'missing', 'three'])
        with self.assertRaisesRegexp(ValueError, 'missing.*one.*two'):
            ServePrediction.check_request(X, [KEYS.PREDICTION.ID, 'one', 'two', 'three'])

    def test_check_request_fail_nulls(self):
        X = pd.DataFrame(
            [[0, 1, np.nan, 3], [4, 5, 6, np.nan]],
            columns=[KEYS.PREDICTION.ID, 'one', 'two', 'three'])
        with self.assertRaisesRegexp(ValueError, 'null.*two.*three'):
            ServePrediction.check_request(X, [KEYS.PREDICTION.ID, 'one', 'two', 'three'])

    def test_check_request_ignore_nulls_pass(self):
        X = pd.DataFrame(
            [[0, 1, np.nan, 3], [4, 5, 6, np.nan]],
            columns=[KEYS.PREDICTION.ID, 'one', 'two', 'three'])
        # no error shoudl be raised
        ServePrediction.check_request(X, ['one', 'two', 'three'], True)

    def test_check_request_ignore_nulls_no_check(self):
        # check that the computation counting nulls is never performed
        mock_X = mock.Mock()
        # no error shoudl be raised
        ServePrediction.check_request(mock_X, ['one', 'two', 'three'], True)
        mock_X.isnull.assert_not_called()



class TestNumpyEncoder(unittest.TestCase):
    def test_default(self):
        encoder = NumpyEncoder()
        actual_type = type(encoder.default(np.int32(1)))
        expected_type = int
        self.assertIs(actual_type, expected_type)

        actual_type = type(encoder.default(np.float32(1)))
        expected_type = float
        self.assertIs(actual_type, expected_type)

        actual_type = type(encoder.default(np.array([[1]])))
        expected_type = list
        self.assertIs(actual_type, expected_type)

        with self.assertRaises(TypeError):
            actual_type = type(encoder.default(1))
            expected_type = list
            self.assertIs(actual_type, expected_type)

    def test_with_json_dumps(self):
        x = np.array([[np.float32(4.0)], [np.int32(0)]])
        actual = json.dumps(x, cls=NumpyEncoder)
        expected = '[[4.0], [0.0]]'
        self.assertEqual(actual, expected)


class TestModelApp(unittest.TestCase):
    @mock.patch('porter.services.ModelApp._build_app')
    @mock.patch('porter.services.ModelApp.add_service')
    def test_add_services(self, mock_add_service, mock__build_app):
        configs = [object(), object(), object()]
        model_app = ModelApp()
        model_app.add_services(configs[0], configs[1], configs[2])
        expected_calls = [mock.call(obj) for obj in configs]
        mock_add_service.assert_has_calls(expected_calls)


if __name__ == '__main__':
    unittest.main()
