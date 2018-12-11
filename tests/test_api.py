import types
import unittest
from unittest import mock

from porter.api import validate_url


class Test(unittest.TestCase):
    def test_validate_url(self):
        self.assertTrue(validate_url('http://foo/bar'))
        self.assertTrue(validate_url('http://foo.com/bar'))
        self.assertTrue(validate_url('http://127.0.0.1:8000/bar/baz/'))
        # missing schema
        self.assertFalse(validate_url('foo.com/bar'))
        self.assertFalse(validate_url('127.0.0.1:8000/bar/baz/'))


if __name__ == '__main__':
    unittest.main()
