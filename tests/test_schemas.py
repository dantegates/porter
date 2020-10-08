import unittest

import porter.schemas as sc

class TestSchemas(unittest.TestCase):
    """Test schemas in schemas.py"""
    def test_app_meta(self):
        # arbitrary dict, anything goes
        sc.app_meta.validate(dict())
        sc.app_meta.validate(dict(a=1, b=3.14, c='test'))

    def test_error_body(self):
        sc.error_body.validate(dict(
            name='BadRequest',
            messages=['The browser (or proxy) sent a request that this server '
                      'could not understand.']
        ))
        with self.assertRaisesRegex(ValueError, 'data.messages must be array'):
            sc.error_body.validate(dict(
                name='BadRequest',
                messages='The browser (or proxy) sent a request that this server '
                          'could not understand.'
            ))

    def test_error_messages(self):
        sc.error_messages.validate([
            'this is an error message',
            'and so is this', ])
        sc.error_messages.validate([])
        with self.assertRaisesRegex(ValueError, 'data must be array'):
            sc.error_messages.validate(dict(messages=['fail']))

    def test_error_name(self):
        self.assertIsInstance(sc.error_name, sc.String)

    def test_error_traceback(self):
        self.assertIsInstance(sc.error_name, sc.String)

    def test_generic_error(self):
        # TODO: generic_error exported but not used anywhere?
        # https://github.com/CadentTech/porter/issues/31
        pass

    def test_health_check(self):
        # borrowed from TestHealthChecks.test_make_alive_ready_response_is_ready
        valid_response = {
            'request_id': '123',
            'porter_version': '0.15.3',
            'deployed_on': '2020-04-16T15:53:34.269254',
            'app_meta': {'foo': 1},
            'services': {
                0: {
                    'model_context': {
                        'model_name': 'svc0',
                        'api_version': '0',
                        'model_meta': {'k': 0, 'v': 1}
                    },
                    'status': 'READY',
                    'endpoint': '/0'
                },
                1: {
                    'model_context': {
                        'model_name': 'svc1',
                        'api_version': '1',
                        'model_meta': {'k': 1, 'v': 2}
                    },
                    'status': 'READY',
                    'endpoint': '/1'
                },
                2: {
                    'model_context': {
                        'model_name': 'svc2',
                        'api_version': '2',
                        'model_meta': {'k': 2, 'v': 3}
                    },
                    'status': 'READY',
                    'endpoint': '/2'
                }
            }
        }
        sc.health_check.validate(valid_response)
        # now try an invalid response
        invalid_response = {
            'request_id': 123,
            'porter_version': '0.15.3',
            'deployed_on': '2020-04-16T15:53:34.269254',
            'app_meta': {'foo': 1},
        }
        with self.assertRaisesRegex(ValueError, 'data must contain'):
            sc.health_check.validate(invalid_response)

    def test_model_context(self):
        valid_context = {
            'model_name': 'a-model',
            'api_version': '1',
            'model_meta': {
                1: '2',
                '3': 4
            }
        }
        sc.model_context.validate(valid_context)
        # now try an invalid context
        invalid_context = {
            'model_name': 'a-model',
            'api_version': 1, # fail here
            'model_meta': {
                1: '2',
                '3': 4
            }
        }
        with self.assertRaisesRegex(ValueError, "data.api_version must be string"):
            sc.model_context.validate(invalid_context)

    def test_model_context_error(self):
        valid_model_context_error = {
            'request_id': '123',
            'error': {
                'name': 'Exception',
                'messages': ('foo bar baz',),
                'traceback': 'Exception: foo bar baz',
                'user_data': {'foo': 1}
            },
            'model_context': {
                'model_name': 'a-model',
                'api_version': '1',
                'model_meta': {
                    1: '2',
                    '3': 4
                }
            }

        }
        sc.model_context_error.validate(valid_model_context_error)
        # now try an invalid error
        invalid_model_context_error = {
            'request_id': '123',
            'error': {
                'name': 'Exception',
                'messages': 'foo bar baz', # fail here
                'traceback': 'Exception: foo bar baz',
                'user_data': {'foo': 1}
            },
            'model_context': {
                'model_name': 'a-model',
                'api_version': '1',
                'model_meta': {
                    1: '2',
                    '3': 4
                }
            }

        }
        with self.assertRaisesRegex(ValueError, "data.error.messages must be array"):
            sc.model_context_error.validate(invalid_model_context_error)

    def test_model_meta(self):
        # arbitrary dict, anything goes
        sc.model_meta.validate(dict())
        sc.model_meta.validate(dict(a=1, b=3.14, c='test'))

    def test_request_id(self):
        self.assertIsInstance(sc.request_id, sc.String)
