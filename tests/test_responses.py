import re
import unittest

from porter.constants import KEYS
from porter.responses import _make_error_payload, _make_prediction_payload, _is_ready


class TestFunctions(unittest.TestCase):
    def test__make_prediction_payload(self):
        actual = _make_prediction_payload(123, [1, 2, 3], [10.0, 11.0, 12.0])
        expected = {
            KEYS.PREDICTION.MODEL_ID: 123,
            KEYS.PREDICTION.PREDICTIONS: [
                {KEYS.PREDICTION.ID: 1, KEYS.PREDICTION.PREDICTION: 10.0},
                {KEYS.PREDICTION.ID: 2, KEYS.PREDICTION.PREDICTION: 11.0},
                {KEYS.PREDICTION.ID: 3, KEYS.PREDICTION.PREDICTION: 12.0}
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
            KEYS.ERROR.ERROR: 'Exception',
            KEYS.ERROR.MESSAGE: ('foo bar baz',),
            KEYS.ERROR.TRACEBACK: ('.*'
                          'line [0-9]*, in test__make_error_payload\n'
                          '    raise error\n'
                          'Exception: foo bar baz.*')
        }
        self.assertEqual(actual[KEYS.ERROR.ERROR], expected[KEYS.ERROR.ERROR])
        self.assertEqual(actual[KEYS.ERROR.MESSAGE], expected[KEYS.ERROR.MESSAGE])
        print(actual[KEYS.ERROR.TRACEBACK])
        self.assertTrue(re.search(expected[KEYS.ERROR.TRACEBACK], actual[KEYS.ERROR.TRACEBACK]))

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
