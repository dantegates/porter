import re
import unittest
from unittest import mock

from porter.exceptions import PredictionError
from porter.responses import (_init_model_context, _is_ready,
                              make_alive_response,
                              make_batch_prediction_response,
                              make_error_response, make_middleware_response,
                              make_prediction_response, make_ready_response)


class Test(unittest.TestCase):
    def test_make_batch_prediction_response(self):
        actual = make_batch_prediction_response('a-model', '1', {1: '2', '3': 4}, [1, 2, 3], [10.0, 11.0, 12.0])
        expected = {
            'model_name': 'a-model',
            'api_version': '1',
            1: '2',
            '3': 4,
            'predictions': [
                {'id': 1, 'prediction': 10.0},
                {'id': 2, 'prediction': 11.0},
                {'id': 3, 'prediction': 12.0}
            ]
        }
        self.assertEqual(actual.data, expected)
        self.assertIsNone(actual.status_code)

    def test_make_prediction_response(self):
        actual = make_prediction_response('a-model', '1', {1: '2', '3': 4}, 1, 10.0)
        expected = {
            'model_name': 'a-model',
            'api_version': '1',
            1: '2',
            '3': 4,
            'predictions': {'id': 1, 'prediction': 10.0}
        }
        self.assertEqual(actual.data, expected)
        self.assertIsNone(actual.status_code)


@mock.patch('porter.responses.api.request_id', lambda: 123)
@mock.patch('porter.responses.api.request_json', lambda *args, **kwargs: {'foo': 1})
class TestErrorResponses(unittest.TestCase):
    @mock.patch('porter.responses.cf.return_message_on_error', True)
    @mock.patch('porter.responses.cf.return_traceback_on_error', True)
    @mock.patch('porter.responses.cf.return_user_data_on_error', False)
    @mock.patch('porter.responses.cf.return_request_id_on_error', True)
    def test_make_error_response_non_porter_error(self):
        error = Exception('foo bar baz')
        try:
            raise error
        except Exception:
            actual = make_error_response(error)
            actual_data = actual.data
            actual_status_code = actual.status_code
        expected = {
            'request_id': 123,
            'error': {
                'name': 'Exception',
                'messages': ('foo bar baz',),
                'traceback': ('.*'
                              'line [0-9]*, in test_make_error_response_non_porter_error\n'
                              '    raise error\n'
                              'Exception: foo bar baz.*'),
                'user_data': {'foo': 1}
            }
        }
        self.assertEqual(actual_data['error']['name'], expected['error']['name'])
        self.assertEqual(actual_data['request_id'], expected['request_id'])
        self.assertEqual(actual_data['error']['messages'], expected['error']['messages'])
        self.assertTrue(re.search(expected['error']['traceback'], actual_data['error']['traceback']))

    @mock.patch('porter.responses.cf.return_message_on_error', True)
    @mock.patch('porter.responses.cf.return_traceback_on_error', True)
    @mock.patch('porter.responses.cf.return_user_data_on_error', True)
    @mock.patch('porter.responses.cf.return_request_id_on_error', True)
    def test_make_error_response_porter_error(self):
        error = PredictionError('foo bar baz')
        error.update_model_context(model_name='M', api_version='V',
            model_meta={1: '1', '2': 2})
        try:
            raise error
        except Exception:
            actual = make_error_response(error)
            actual_data = actual.data
            actual_status_code = actual.status_code
        expected = {
            'model_context': {
                'model_name': 'M',
                'api_version': 'V',
                1: '1',
                '2': 2,
            },
            'request_id': 123,
            'error': {
                'name': 'PredictionError',
                'messages': ('foo bar baz',),
                'traceback': ('.*'
                              'line [0-9]*, in test_make_error_response_porter_error\n'
                              '    raise error\n'
                              'porter.exceptions.PredictionError: foo bar baz.*'),
                'user_data': {'foo': 1}
            }
        }
        self.assertEqual(actual_data['model_context'], expected['model_context'])
        self.assertEqual(actual_data['error']['name'], expected['error']['name'])
        self.assertEqual(actual_data['request_id'], expected['request_id'])
        self.assertEqual(actual_data['error']['messages'], expected['error']['messages'])
        self.assertTrue(re.search(expected['error']['traceback'], actual_data['error']['traceback']))
        self.assertEqual(actual_data['error']['user_data'], expected['error']['user_data'])

    @mock.patch('porter.responses.cf.return_message_on_error', True)
    @mock.patch('porter.responses.cf.return_traceback_on_error', True)
    @mock.patch('porter.responses.cf.return_user_data_on_error', False)
    @mock.patch('porter.responses.cf.return_request_id_on_error', True)
    def test_make_error_response_custom_response_keys_no_user_data(self):
        error = Exception('foo bar baz')
        try:
            raise error
        except Exception:
            actual = make_error_response(error)
            actual_data = actual.data
            actual_status_code = actual.status_code
        expected = {
            'request_id': 123,
            'error': {
                'name': 'Exception',
                'messages': ('foo bar baz',),
                'traceback': ('.*'
                              'line [0-9]*, in test_make_error_response_custom_response_keys_no_user_data\n'
                              '    raise error\n'
                              'Exception: foo bar baz.*')
            }
        }
        self.assertEqual(actual_data['error']['name'], expected['error']['name'])
        self.assertEqual(actual_data['request_id'], expected['request_id'])
        self.assertEqual(actual_data['error']['messages'], expected['error']['messages'])
        self.assertTrue(re.search(expected['error']['traceback'], actual_data['error']['traceback']))
        self.assertNotIn('user_data', actual_data['error'])

    @mock.patch('porter.responses.cf.return_message_on_error', False)
    @mock.patch('porter.responses.cf.return_traceback_on_error', False)
    @mock.patch('porter.responses.cf.return_user_data_on_error', False)
    @mock.patch('porter.responses.cf.return_request_id_on_error', False)
    def test_make_error_response_custom_response_keys_name_only(self):
        error = Exception('foo bar baz')
        try:
            raise error
        except Exception:
            actual = make_error_response(error)
            actual_data = actual.data
            actual_status_code = actual.status_code
        expected = {
            'error': {
                'name': 'Exception',
            }
        }
        self.assertEqual(actual_data['error']['name'], expected['error']['name'])
        self.assertNotIn('request_id', actual_data)
        self.assertNotIn('messages', actual_data['error'])
        self.assertNotIn('traceback', actual_data['error'])
        self.assertNotIn('user_data', actual_data['error'])

    @mock.patch('porter.responses.cf.return_message_on_error', True)
    @mock.patch('porter.responses.cf.return_traceback_on_error', False)
    @mock.patch('porter.responses.cf.return_user_data_on_error', False)
    @mock.patch('porter.responses.cf.return_request_id_on_error', True)
    def test_make_error_response_custom_response_keys_name_and_messages(self):
        error = Exception('foo bar baz')
        try:
            raise error
        except Exception:
            actual = make_error_response(error)
            actual_data = actual.data
            actual_status_code = actual.status_code
        expected = {
            'request_id': 123,
            'error': {
                'name': 'Exception',
                'messages': ('foo bar baz',),
            }
        }
        self.assertEqual(actual_data['error']['name'], expected['error']['name'])
        self.assertEqual(actual_data['request_id'], expected['request_id'])
        self.assertEqual(actual_data['error']['messages'], expected['error']['messages'])
        self.assertNotIn('traceback', actual_data['error'])
        self.assertNotIn('user_data', actual_data['error'])

    def test__is_ready(self):
        app_state = {
            'services': {
                'model1': {'status': 'READY'},
                'model2': {'status': 'READY'},
            }
        }
        ready = _is_ready(app_state)
        self.assertTrue(ready)

    def test__is_ready_not_ready1(self):
        app_state = {
            'services': {}
        }
        ready = _is_ready(app_state)
        self.assertFalse(ready)

    def test__is_ready_not_ready2(self):
        app_state = {
            'services': {
                'model1': {'status': 'READY'},
                'model2': {'status': 'NO'},
            }
        }
        ready = _is_ready(app_state)
        self.assertFalse(ready)


if __name__ == '__main__':
    unittest.main()
