"""
Tests for the `.app` attribute belonging to an instance of `ipa.ModelService`.
"""


import json
import unittest

import flask

from ipa.services import ModelApp


class TestAppErrorHandling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # DO NOT set app.testing = True here
        # doing so *disables* error handling in the application and instead
        # passes errors on to the test client (in our case, instances of
        # unittest.TestCase).
        # In this class we actually want to test the applications error handling
        # and thus do not set this attribute.
        # See, http://flask.pocoo.org/docs/0.12/api/#flask.Flask.test_client
        app = ModelApp().app
        @app.route('/test-error-handling/', methods=['POST'])
        def test_error():
            flask.request.get_json(force=True)
            raise Exception('exceptional testing of exceptions')
        cls.app = app.test_client()

    def test_bad_request(self):
        # note data is unreadable JSON, thus a BadRequest
        resp = self.app.post('/test-error-handling/', data='')
        self.validate_error_response(resp, 'BadRequest', 400)

    def test_not_found(self):
        resp = self.app.get('/not-found/')
        self.validate_error_response(resp, 'NotFound', 404)

    def test_method_not_allowed(self):
        resp = self.app.get('/test-error-handling/')
        self.validate_error_response(resp, 'MethodNotAllowed', 405)

    def test_internal_server_error(self):
        resp = self.app.post('/test-error-handling/', data='{}')
        self.validate_error_response(resp, 'Exception', 500, 'exceptional', 'raise Exception')

    def validate_error_response(self, response, error, status_code, message_substr=None,
                                traceback_substr=None):
        data = json.loads(response.data)
        self.assertEqual(data['error'], error)
        self.assertEqual(response.status_code, status_code)
        # even if we don't check the values of message or traceback at least
        # make sure that these keys are returned to the user.
        self.assertIn('message', data)
        self.assertIn('traceback', data)
        if not message_substr is None:
            self.assertIn(message_substr, data['message'])
        if not traceback_substr is None:
            self.assertIn(traceback_substr, data['traceback'])


if __name__ == '__main__':
    unittest.main()
