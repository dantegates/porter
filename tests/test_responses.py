import json
import unittest

from porter.responses import PredictionPayload, ErrorPayload


class TestPredictionPayload(unittest.TestCase):
    def test_constructor(self):
        actual = PredictionPayload(123, [1, 2, 3], [10.0, 11.0, 12.0])
        expected = {
            'model_id': 123,
            'predictions': [
                {1: 10.0},
                {2: 11.0},
                {3: 12.0}
            ]
        }
        self.assertItemsEqual(actual, expected)

    def test__init_payload(self):
        actual = PredictionPayload(123, [1, 2, 3], [10.0, 11.0, 12.0])
        expected = {
            'model_id': 123,
            'predictions': [
                {"id": 1, "prediction": 10.0},
                {"id": 2, "prediction": 11.0},
                {"id": 3, "prediction": 12.0}
            ]
        }
        self.assertItemsEqual(actual, expected)

    def test_json_compatability(self):
        response = PredictionPayload(123, [1, 2, 3], [10.0, 11.0, 12.0])
        actual_json = json.dumps(response)
        # rather than comparing strings (which would include using OrderedDicts
        # and writing out the JSON as a str) we simply decoded the encoded JSON.
        #
        # If PredictionPayload is incompatible with `json.dumps` either an error
        # will be raised or the expected decoded object will not match what
        # actually happened.
        # In either event the test will fail.
        actual_decoded_json = json.loads(actual_json)
        expected_decoded_json = {
            'model_id': 123,
            'predictions': [
                {"id": 1, "prediction": 10.0},
                {"id": 2, "prediction": 11.0},
                {"id": 3, "prediction": 12.0}
            ]
        }
        self.assertItemsEqual(actual_decoded_json, expected_decoded_json)


class TestErrorPayload(unittest.TestCase):
    def test_constructor(self):
        error = Exception('foo bar baz')
        try:
            raise error
        except:
            actual = ErrorPayload(error)
        expected = {
            'error': 'Exception',
            'message': 'foo bar baz',
            'traceback': 'raise error'
        }

    def test__init_payload(self):
        error = Exception('foo bar baz')
        try:
            raise error
        except:
            actual = ErrorPayload._init_payload(error)
        expected = {
            'error': 'Exception',
            'message': 'foo bar baz',
            'traceback': 'raise error'
        }


    def test_json_compatability(self):
        error = Exception('foo bar baz')
        try:
            raise error
        except:
            response = ErrorPayload._init_payload(error)
        actual_json = json.dumps(response)
        # rather than comparing strings (which would include using OrderedDicts
        # and writing out the JSON as a str) we simply decoded the encoded JSON.
        #
        # If PredictionPayload is incompatible with `json.dumps` either an error
        # will be raised or the expected decoded object will not match what
        # actually happened.
        # In either event the test will fail.
        actual_decoded_json = json.loads(actual_json)
        expected_decoded_json = {
            'error': 'Exception',
            'message': 'foo bar baz',
            'traceback': 'raise error'
        }
        self.assertItemsEqual(actual_decoded_json, expected_decoded_json)


if __name__ == '__main__':
    unittest.main()
