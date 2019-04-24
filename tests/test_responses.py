import re
import unittest

from porter.exceptions import PredictionError
from porter.responses import (_is_ready, _make_batch_prediction_payload,
                              _make_error_payload,
                              _make_single_prediction_payload)


class TestFunctions(unittest.TestCase):
    def test__make_batch_prediction_payload(self):
        actual = _make_batch_prediction_payload('a-model', '1', {1: '2', '3': 4}, [1, 2, 3], [10.0, 11.0, 12.0])
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
        self.assertEqual(actual, expected)

    def test__make_single_prediction_payload(self):
        actual = _make_single_prediction_payload('a-model', '1', {1: '2', '3': 4}, [1], [10.0])
        expected = {
            'model_name': 'a-model',
            'api_version': '1',
            1: '2',
            '3': 4,
            'predictions': {'id': 1, 'prediction': 10.0}
        }
        self.assertEqual(actual, expected)

    def test__make_error_payload_non_porter_error(self):
        error = Exception('foo bar baz')
        try:
            raise error
        except Exception:
            actual = _make_error_payload(error, 123, user_data='foo',
                include_message=True, include_traceback=True,
                include_user_data=True)
        expected = {
            'error': {
                'name': 'Exception',
                'request_id': 123,
                'messages': ('foo bar baz',),
                'traceback': ('.*'
                              'line [0-9]*, in test__make_error_payload_non_porter_error\n'
                              '    raise error\n'
                              'Exception: foo bar baz.*'),
                'user_data': 'foo'
            }
        }
        self.assertEqual(actual['error']['name'], expected['error']['name'])
        self.assertEqual(actual['error']['request_id'], expected['error']['request_id'])
        self.assertEqual(actual['error']['messages'], expected['error']['messages'])
        self.assertTrue(re.search(expected['error']['traceback'], actual['error']['traceback']))

    def test__make_error_payload_porter_error(self):
        error = PredictionError('foo bar baz')
        error.update_model_context(model_name='M', api_version='V',
            model_meta={1: '1', '2': 2})
        try:
            raise error
        except Exception:
            actual = _make_error_payload(error, 123, user_data='foo',
                include_message=True, include_traceback=True,
                include_user_data=True)
        expected = {
            'model_name': 'M',
            'api_version': 'V',
            1: '1',
            '2': 2,
            'error': {
                'name': 'PredictionError',
                'request_id': 123,
                'messages': ('foo bar baz',),
                'traceback': ('.*'
                              'line [0-9]*, in test__make_error_payload_porter_error\n'
                              '    raise error\n'
                              'porter.exceptions.PredictionError: foo bar baz.*'),
                'user_data': 'foo'
            }
        }
        self.assertEqual(actual['error']['name'], expected['error']['name'])
        self.assertEqual(actual['error']['request_id'], expected['error']['request_id'])
        self.assertEqual(actual['error']['messages'], expected['error']['messages'])
        self.assertTrue(re.search(expected['error']['traceback'], actual['error']['traceback']))
        self.assertEqual(actual['error']['user_data'], expected['error']['user_data'])

    def test__make_error_payload_custom_response_keys_no_user_data(self):
        error = Exception('foo bar baz')
        try:
            raise error
        except Exception:
            actual = _make_error_payload(error, 123, user_data='foo',
                include_message=True, include_traceback=True,
                include_user_data=False)
        expected = {
            'error': {
                'name': 'Exception',
                'request_id': 123,
                'messages': ('foo bar baz',),
                'traceback': ('.*'
                              'line [0-9]*, in test__make_error_payload_custom_response_keys_no_user_data\n'
                              '    raise error\n'
                              'Exception: foo bar baz.*')
            }
        }
        self.assertEqual(actual['error']['name'], expected['error']['name'])
        self.assertEqual(actual['error']['request_id'], expected['error']['request_id'])
        self.assertEqual(actual['error']['messages'], expected['error']['messages'])
        self.assertTrue(re.search(expected['error']['traceback'], actual['error']['traceback']))
        self.assertNotIn('user_data', actual['error'])

    def test__make_error_payload_custom_response_keys_name_only(self):
        error = Exception('foo bar baz')
        try:
            raise error
        except Exception:
            actual = _make_error_payload(error, 123, user_data='foo',
                include_message=False, include_traceback=False,
                include_user_data=False)
        expected = {
            'error': {
                'name': 'Exception',
                'request_id': 123,
            }
        }
        self.assertEqual(actual['error']['name'], expected['error']['name'])
        self.assertEqual(actual['error']['request_id'], expected['error']['request_id'])
        self.assertNotIn('messages', actual['error'])
        self.assertNotIn('traceback', actual['error'])
        self.assertNotIn('user_data', actual['error'])

    def test__make_error_payload_custom_response_keys_name_and_messages(self):
        error = Exception('foo bar baz')
        try:
            raise error
        except Exception:
            actual = _make_error_payload(error, 123, user_data='foo',
                include_message=True, include_traceback=False,
                include_user_data=False)
        expected = {
            'error': {
                'name': 'Exception',
                'request_id': 123,
                'messages': ('foo bar baz',),
            }
        }
        self.assertEqual(actual['error']['name'], expected['error']['name'])
        self.assertEqual(actual['error']['request_id'], expected['error']['request_id'])
        self.assertEqual(actual['error']['messages'], expected['error']['messages'])
        self.assertNotIn('traceback', actual['error'])
        self.assertNotIn('user_data', actual['error'])

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
