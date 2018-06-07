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
        self.assertItemsEqual(actual, expected)

    def test__make_error_payload(self):
        error = Exception('foo bar baz')
        try:
            raise error
        except Exception:
            actual = _make_error_payload(error)
        expected = {
            'error': 'Exception',
            'message': 'foo bar baz',
            'traceback': 'raise error'
        }
        self.assertItemsEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
