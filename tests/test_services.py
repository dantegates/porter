import mock
import unittest

import numpy as np
import pandas as pd
from porter.services import (_ID_KEY, ModelApp, check_request,
                          serve_error_message, serve_prediction)


class TestFuntions(unittest.TestCase):
    def test_check_request_pass(self):
        # no error should be raised
        X = pd.DataFrame(
            [[0, 1, 2, 3], [4, 5, 6, 7]],
            columns=[_ID_KEY, 'one', 'two', 'three'])
        check_request(X, ['one', 'two', 'three'])

    def test_check_request_fail_missing_id(self):
        X = pd.DataFrame(
            [[0, 1, 2, 3], [4, 5, 6, 7]],
            columns=['missing', 'one', 'two', 'three'])
        with self.assertRaises(ValueError):
            check_request(X, ['one', 'two', 'three'])

    def test_check_request_fail_missing_id_column(self):
        X = pd.DataFrame(
            [[0, 1, 2, 3], [4, 5, 6, 7]],
            columns=['missing', 'one', 'two', 'three'])
        with self.assertRaisesRegexp(ValueError, 'missing.*id'):
            check_request(X, ['one', 'two', 'three'])

    def test_check_request_fail_missing_input_columns(self):
        X = pd.DataFrame(
            [[0, 1, 2, 3], [4, 5, 6, 7]],
            columns=[_ID_KEY, 'missing', 'missing', 'three'])
        with self.assertRaisesRegexp(ValueError, 'missing.*one.*two'):
            check_request(X, ['one', 'two', 'three'])

    def test_check_request_fail_nulls(self):
        X = pd.DataFrame(
            [[0, 1, np.nan, 3], [4, 5, 6, np.nan]],
            columns=[_ID_KEY, 'one', 'two', 'three'])
        with self.assertRaisesRegexp(ValueError, 'null.*two.*three'):
            check_request(X, ['one', 'two', 'three'])

    def test_check_request_ignore_nulls_pass(self):
        X = pd.DataFrame(
            [[0, 1, np.nan, 3], [4, 5, 6, np.nan]],
            columns=[_ID_KEY, 'one', 'two', 'three'])
        # no error shoudl be raised
        check_request(X, ['one', 'two', 'three'], True)

    def test_check_request_ignore_nulls_no_check(self):
        # check that the computation counting nulls is never performed
        mock_X = mock.Mock()
        # no error shoudl be raised
        check_request(mock_X, ['one', 'two', 'three'], True)
        mock_X.isnull.assert_not_called()

    def test_serve_prediction(self):
        pass

    def test_serve_error_message(self):
        pass


class TestModelService(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
