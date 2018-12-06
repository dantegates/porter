import functools
import types
import unittest
from unittest import mock

from porter.api import validate_url, cache_during_request


class Test(unittest.TestCase):
    def test_validate_url(self):
        self.assertTrue(validate_url('http://foo/bar'))
        self.assertTrue(validate_url('http://foo.com/bar'))
        self.assertTrue(validate_url('http://127.0.0.1:8000/bar/baz/'))
        # missing schema
        self.assertFalse(validate_url('foo.com/bar'))
        self.assertFalse(validate_url('127.0.0.1:8000/bar/baz/'))

    def test_cache_during_request(self):
        @cache_during_request
        def f():
            return 1
        
        @cache_during_request
        def g(x=[10]):
            x[0] += 1
            return x[0]

        with mock.patch('porter.api.flask.g', types.SimpleNamespace()):
            self.assertEqual(f(), 1)
            self.assertEqual(g(), 11)
            self.assertEqual(g(), 11)

        with mock.patch('porter.api.flask.g', types.SimpleNamespace()):
            self.assertEqual(f(), 1)
            self.assertEqual(g(), 12)
            self.assertEqual(g(), 12)


if __name__ == '__main__':
    unittest.main()
