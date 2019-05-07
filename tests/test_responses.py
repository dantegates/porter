import re
import unittest
from unittest import mock

from porter import __version__ as VERSION
from porter import constants as cn
from porter.exceptions import PredictionError
from porter.responses import (_build_app_state, _init_model_context, _is_ready,
                              make_alive_response,
                              make_batch_prediction_response,
                              make_error_response, make_middleware_response,
                              make_prediction_response, make_ready_response)


class Test(unittest.TestCase):
    def test_make_batch_prediction_response(self):
        # on setting name after instantiation see
        # https://docs.python.org/3/library/unittest.mock.html#mock-names-and-the-name-attribute
        mock_model_service = mock.Mock(
            api_version='1', meta={1: '2', '3': 4})
        mock_model_service.configure_mock(name='a-model')
        actual = make_batch_prediction_response(mock_model_service, [1, 2, 3], [10.0, 11.0, 12.0])
        expected = {
            'model_context': {
                'model_name': 'a-model',
                'api_version': '1',
                'model_meta': {
                    1: '2',
                    '3': 4,
                }
            },
            'predictions': [
                {'id': 1, 'prediction': 10.0},
                {'id': 2, 'prediction': 11.0},
                {'id': 3, 'prediction': 12.0}
            ]
        }
        self.assertEqual(actual.data, expected)
        self.assertIsNone(actual.status_code)

    def test_make_prediction_response(self):
        # on setting name after instantiation see
        # https://docs.python.org/3/library/unittest.mock.html#mock-names-and-the-name-attribute
        mock_model_service = mock.Mock(
            api_version='1', meta={1: '2', '3': 4})
        mock_model_service.configure_mock(name='a-model')
        actual = make_prediction_response(mock_model_service, 1, 10.0)
        expected = {
            'model_context': {
                'model_name': 'a-model',
                'api_version': '1',
                'model_meta': {
                    1: '2',
                    '3': 4
                }
            },
            'predictions': {'id': 1, 'prediction': 10.0}
        }
        self.assertEqual(actual.data, expected)
        self.assertIsNone(actual.status_code)

    @mock.patch('porter.responses.cf.return_request_id_with_prediction', True)
    @mock.patch('porter.responses.api.request_id', lambda: 123)
    def test_make_batch_prediction_response_with_request_id(self):
        # on setting name after instantiation see
        # https://docs.python.org/3/library/unittest.mock.html#mock-names-and-the-name-attribute
        mock_model_service = mock.Mock(
            api_version='1', meta={1: '2', '3': 4})
        mock_model_service.configure_mock(name='a-model')
        actual = make_batch_prediction_response(mock_model_service, [1, 2, 3], [10.0, 11.0, 12.0])
        expected = {
            'request_id': 123,
            'model_context': {
                'model_name': 'a-model',
                'api_version': '1',
                'model_meta': {
                    1: '2',
                    '3': 4,
                }
            },
            'predictions': [
                {'id': 1, 'prediction': 10.0},
                {'id': 2, 'prediction': 11.0},
                {'id': 3, 'prediction': 12.0}
            ]
        }
        self.assertEqual(actual.data, expected)
        self.assertIsNone(actual.status_code)

    @mock.patch('porter.responses.cf.return_request_id_with_prediction', True)
    @mock.patch('porter.responses.api.request_id', lambda: 123)
    def test_make_prediction_response_with_request_id(self):
        # on setting name after instantiation see
        # https://docs.python.org/3/library/unittest.mock.html#mock-names-and-the-name-attribute
        mock_model_service = mock.Mock(
            api_version='1', meta={1: '2', '3': 4})
        mock_model_service.configure_mock(name='a-model')
        actual = make_prediction_response(mock_model_service, 1, 10.0)
        expected = {
            'request_id': 123,
            'model_context': {
                'model_name': 'a-model',
                'api_version': '1',
                'model_meta': {
                    1: '2',
                    '3': 4
                }
            },
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
        # on setting name after instantiation see
        # https://docs.python.org/3/library/unittest.mock.html#mock-names-and-the-name-attribute
        error = PredictionError('foo bar baz')
        mock_model_service = mock.Mock(
            api_version='V', meta={1: '1', '2': 2},
            id='M:V')
        mock_model_service.configure_mock(name='M')
        error.update_model_context(mock_model_service)
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
                'model_meta': {
                    1: '1',
                    '2': 2
                }
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


class TestHealthChecks(unittest.TestCase):
    def test__build_app_state(self):
        mock_app = mock.Mock(meta={'foo': 1})
        mock_app.meta
        mock_app._services = [mock.Mock(status='READY' if i % 2 else 'NOTREADY',
                                        api_version=str(i),
                                        meta={'k': i, 'v': i+1},
                                        id=i,
                                        endpoint=f'/{i}')
                              for i in range(3)]
        _ = [m.configure_mock(name=f'svc{i}') for i, m in enumerate(mock_app._services)]
        actual = _build_app_state(mock_app)
        expected = {
            'porter_version': VERSION,
            'deployed_on': cn.HEALTH_CHECK_VALUES.DEPLOYED_ON,
            'app_meta': {'foo': 1},
            'services': {
                0: {
                    'model_context': {
                        'model_name': 'svc0',
                        'api_version': '0',
                        'model_meta': {'k': 0, 'v': 1}
                    },
                    'status': 'READY',
                    'endpoint': '/0'
                },
                1: {
                    'model_context': {
                        'model_name': 'svc1',
                        'api_version': '1',
                        'model_meta': {'k': 1, 'v': 2}
                    },
                    'status': 'NOTREADY',
                    'endpoint': '/1'
                },
                2: {
                    'model_context': {
                        'model_name': 'svc2',
                        'api_version': '2',
                        'model_meta': {'k': 2, 'v': 3}
                    },
                    'status': 'READY',
                    'endpoint': '/2'
                }
            }
        }
        self.assertCountEqual(actual, expected)
        self.assertCountEqual(actual['app_meta'], expected['app_meta'])
        self.assertCountEqual(actual['services'], expected['services'])
        for key in actual['services']:
            self.assertCountEqual(actual['services'][key], expected['services'][key])
        for key in actual['services']:
            self.assertEqual(actual['services'][key]['model_context'], expected['services'][key]['model_context'])

    def test__build_app_state_no_services(self):
        mock_app = mock.Mock(meta={'foo': 1}, _services=[])
        mock_app.meta
        actual = _build_app_state(mock_app)
        expected = {
            'porter_version': VERSION,
            'deployed_on': cn.HEALTH_CHECK_VALUES.DEPLOYED_ON,
            'app_meta': {'foo': 1},
            'services': {}
        }
        self.assertCountEqual(actual, expected)
        self.assertCountEqual(actual['app_meta'], expected['app_meta'])
        self.assertCountEqual(actual['services'], expected['services'])
        for key in actual['services']:
            self.assertCountEqual(actual['services'][key], expected['services'][key])

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
