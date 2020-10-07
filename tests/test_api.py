import gzip
import json

import types
import unittest
from unittest import mock

from porter import api
import werkzeug.exceptions as werkzeug_exc


class test_request:
    """Substitute for ``flask.request`` with data, content_encoding, headers, get_data(), and get_json()."""
    def __init__(self, data, content_encoding=None, accept_encoding=''):
        self.data, self.content_encoding = data, content_encoding
        self.headers = {'Accept-Encoding': accept_encoding}
    def get_data(self):
        return self.data
    def get_json(self, **kw):
        return json.loads(self.data.decode('utf-8'))

class test_response:
    """Substitute for ``flask.Response`` with data and headers."""
    def __init__(self, data):
        self.data, self.headers = data, {}

class TestDecodeRequest(unittest.TestCase):

    """Test request decoding."""

    def setUp(self):
        self.valid_bytes = b'{"a": 1, "b": 2.3}'
        self.valid_dict = json.loads(self.valid_bytes)
        self.valid_gzip = gzip.compress(self.valid_bytes)

    def test_request_json(self):
        """Test well-formed request: data matches stated encoding."""
        with mock.patch('flask.request', test_request(self.valid_bytes, None)):
            self.assertEqual(self.valid_dict, api.request_json())
        with mock.patch('flask.request', test_request(self.valid_bytes, 'identity')):
            self.assertEqual(self.valid_dict, api.request_json())
        with mock.patch('flask.request', test_request(self.valid_gzip, 'gzip')):
            self.assertEqual(self.valid_dict, api.request_json())

    def test_request_json_bad_request(self):
        """Test bad data or mismatched data vs encoding."""
        with mock.patch('flask.request', test_request(b'{"invalid_json": true', None)):
            with self.assertRaises(werkzeug_exc.BadRequest):
                api.request_json()
        with mock.patch('flask.request', test_request(self.valid_bytes, 'gzip')):
            with self.assertRaises(werkzeug_exc.BadRequest):
                api.request_json()
        with mock.patch('flask.request', test_request(self.valid_gzip, None)):
            with self.assertRaises(werkzeug_exc.BadRequest):
                api.request_json()

    def test_request_json_unsupported(self):
        """Test unsupported encoding."""
        # legal but unsupported
        with mock.patch('flask.request', test_request(self.valid_bytes, 'compress')):
            with self.assertRaises(werkzeug_exc.UnsupportedMediaType):
                api.request_json()
        # illegal
        with mock.patch('flask.request', test_request(self.valid_bytes, 'fake_encoding')):
            with self.assertRaises(werkzeug_exc.UnsupportedMediaType):
                api.request_json()

class TestEncodeResponse(unittest.TestCase):

    """Test response encoding."""

    def setUp(self):
        self.data = b'{"id": 1, "prediction": 0.37}'
        self.response = test_response(self.data)

    def test__gzip_response(self):
        """Test gzip data + added headers."""
        data, response = self.data, self.response
        api._gzip_response(response)
        self.assertEqual(response.headers['Content-Encoding'], 'gzip')
        self.assertEqual(response.headers['Vary'], 'Accept-Encoding')
        self.assertEqual(gzip.decompress(response.data), self.data)

    def test_encode_response_plain(self):
        """Pass thru if no compression requested."""
        _gzip_response = mock.Mock()
        with mock.patch('porter.api._gzip_response', _gzip_response):
            with mock.patch('flask.request', test_request(self.data)):
                api.encode_response(self.response)
                _gzip_response.assert_not_called()

    def test_encode_response_unaccept(self):
        """Pass thru if no acceptable encoding requested."""
        # illegal
        _gzip_response = mock.Mock()
        with mock.patch('porter.api._gzip_response', _gzip_response):
            with mock.patch('flask.request', test_request(self.data, accept_encoding='fake_encoding')):
                api.encode_response(self.response)
                _gzip_response.assert_not_called()

        # legal but unsupported
        _gzip_response = mock.Mock()
        with mock.patch('porter.api._gzip_response', _gzip_response):
            with mock.patch('flask.request', test_request(self.data, accept_encoding='compress')):
                api.encode_response(self.response)
                _gzip_response.assert_not_called()

        # legal and supported but not enabled
        with mock.patch('porter.api._gzip_response', _gzip_response):
            with mock.patch('flask.request', test_request(self.data, accept_encoding='gzip')):
                with mock.patch('porter.config.support_gzip', False):
                    api.encode_response(self.response)
                    _gzip_response.assert_not_called()

    def test_encode_response_gzip(self):
        """Compress with gzip if requested and support is enabled."""
        _gzip_response = mock.Mock()
        with mock.patch('porter.api._gzip_response', _gzip_response):
            with mock.patch('flask.request', test_request(self.data, accept_encoding='gzip')):
                with mock.patch('porter.config.support_gzip', True):
                    api.encode_response(self.response)
                    _gzip_response.assert_called_with(self.response)

class TestValidate(unittest.TestCase):

    def test_validate_url(self):
        self.assertTrue(api.validate_url('http://foo/bar'))
        self.assertTrue(api.validate_url('http://foo.com/bar'))
        self.assertTrue(api.validate_url('http://127.0.0.1:8000/bar/baz/'))
        # missing schema
        self.assertFalse(api.validate_url('foo.com/bar'))
        self.assertFalse(api.validate_url('127.0.0.1:8000/bar/baz/'))


if __name__ == '__main__':
    unittest.main()
