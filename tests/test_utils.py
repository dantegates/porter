import json
import unittest

import numpy as np
from porter.utils import NumpyEncoder


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


if __name__ == '__main__':
    unittest.main()
