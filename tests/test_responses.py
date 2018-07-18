import re
import unittest

from porter.responses import _make_error_payload, _make_prediction_payload, _is_ready


class TestFunctions(unittest.TestCase):
    def test__make_prediction_payload(self):
        actual = _make_prediction_payload('a-model', '1', {1: '2', '3': 4}, [1, 2, 3], [10.0, 11.0, 12.0])
        expected = {
            'model_name': 'a-model',
            'model_version': '1',
            1: '2',
            '3': 4,
            'predictions': [
                {'id': 1, 'prediction': 10.0},
                {'id': 2, 'prediction': 11.0},
                {'id': 3, 'prediction': 12.0}
            ]
        }
        self.assertEqual(actual, expected)

    def test__make_error_payload(self):
        error = Exception('foo bar baz')
        try:
            raise error
        except Exception:
            actual = _make_error_payload(error)
        expected = {
            'error': 'Exception',
            'message': ('foo bar baz',),
            'traceback': ('.*'
                          'line [0-9]*, in test__make_error_payload\n'
                          '    raise error\n'
                          'Exception: foo bar baz.*')
        }
        self.assertEqual(actual['error'], expected['error'])
        self.assertEqual(actual['message'], expected['message'])
        self.assertTrue(re.search(expected['traceback'], actual['traceback']))

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
