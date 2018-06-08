import json
import unittest

from porter.responses import (_make_error_payload, _make_prediction_payload,
                              make_error_response, make_prediction_response)


class TestFunctions(unittest.TestCase):
    def test__make_prediction_payload(self):
        actual = _make_prediction_payload(123, [1, 2, 3], [10.0, 11.0, 12.0])
        expected = {
            'model_id': 123,
            'predictions': [
                {"id": 1, "prediction": 10.0},
                {"id": 2, "prediction": 11.0},
                {"id": 3, "prediction": 12.0}
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
            'traceback': ('Traceback (most recent call last):\n'
                          '  File "/Users/dgates/repos/porter/tests/test_responses.py", '
                          'line 24, in test__make_error_payload\n'
                          '    raise error\n'
                          'Exception: foo bar baz\n')
        }
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
