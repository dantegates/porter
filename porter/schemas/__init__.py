import warnings

from .openapi import (Array, Boolean, Integer, Number, Object,
                      RequestBody, ResponseBody, String)
from .schemas import (error_body, generic_error, health_check, model_context,
                      model_context_error, request_id)


class Contract:
    def __init__(self, method, *, request_schema=None, response_schemas=None,
                 validate_request_data=True, validate_response_data=True,
                 additional_params=None):
        self.method = method.lower()

        if method != 'GET' and validate_request_data and request_schema is None:
            raise ValueError('`request_schema` cannot be None if `validate_request_data` is True.')
        if validate_response_data and response_schemas is None:
            raise ValueError('`response_schemas` cannot be None if `validate_response_data` is True.')

        self.request_schema = request_schema
        self.response_schemas = response_schemas
        self.validate_request_data = validate_request_data

        if validate_response_data:
            warnings.warn('validate_response_data will '
                          'result in returning confusing error messages to the user. '
                          'Use only during development for testing and debugging.')
        self.validate_response_data = validate_response_data
        self.additional_params = {} if additional_params is None else additional_params
        self._response_schema_lookup = {response_schema.status_code: response_schema
                                        for response_schema in response_schemas}

    def fetch_response_schema(self, status_code):
        return self._response_schema_lookup.get(status_code)


__all__ = [
    'Array', 'Boolean', 'Integer', 'Number', 'Object',
    'RequestBody', 'ResponseBody', 'String',
    'error_body', 'generic_error', 'health_check', 'model_context',
    'model_context_error', 'request_id',
]
