import functools
import json
import warnings

import fastjsonschema

from ..api import request_json, request_method
from ..exceptions import InvalidModelInput


# TODO: rename to wrap_validations, use scope to pass contacts to 
# the validation harness instead of attaching to fn and build docs
# in services.py from the api_contracts attribute in the services
def validate(contracts):
    """Decorator to attach API contract definitions to functions. Can be used to
    decorate `flask` routes.

    Args:
        contracts (`list[Contract]`): The contract for the endpoint.
        validate_request (bool): Validate the request data against the schema
            defined by `request`?

    Returns:
        function: The decorated function with the attribute `_contracts` assigned.

    Raises:
        jsonschema.exceptions.InvalidModelInput
    """

    def fn_decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with _ValidationHarness(fn, contracts) as validator:
                return validator.run_fn(*args, **kwargs)
        return wrapper
    return fn_decorator


class _ValidationHarness:
    def __init__(self, fn, contracts):
        self._fn = fn
        self._contracts = {c.method: c for c in contracts}
        self._method = request_method()
        # TODO: raise error if validate request is True but the contract
        #       can't be found or warning?
        self.contract = self._contracts.get(self._method.lower())
        self._fn_response = None

    def __enter__(self):
        if  (self.contract is not None
                and self.contract.validate_request_data
                and self.contract.request_schema is not None):
            data = request_json()
            self._validate_request(self.contract.request_schema, data)
        return self

    def run_fn(self, *args, **kwargs):
        self._fn_response = response = self._fn(*args, **kwargs)
        # we need to store the response on the instance to do validations
        # in __exit__() but we  don't want to return the bound attribute
        return response

    def __exit__(self, *exc):
        data = json.loads(self._fn_response.data)
        response_schema = self.contract.fetch_response_schema(self._fn_response.status_code)
        if (self.contract is not None
                and self.contract.validate_response_data
                and response_schema is not None):
            self._validate_response(response_schema, data)

        return False  # we do not want to ignore any errors that may be in *exc

    def _validate_request(self, request_schema, data):
        try:
            request_schema.obj.validate(data)
        except fastjsonschema.exceptions.JsonSchemaException as err:
            # fastjsonschema raises useful error messsages so we'll reuse them.
            # However, we'll raise a InvalidModelInput to signal that this is a
            # model context error and so that other modules don't need to depend
            # on fastjsonschema
            raise InvalidModelInput(f'{self._method} data failed validation: {err.args[0]}') from err

    def _validate_response(self, response_schema, data):
        try:
            response_schema.obj.validate(data)
        except fastjsonschema.exceptions.JsonSchemaException as err:
            # fastjsonschema raises useful error messsages so we'll reuse them.
            # However, we'll raise a InvalidModelInput to signal that this is a
            # model context error and so that other modules don't need to depend
            # on fastjsonschema
            raise Exception(f'{self._method} return data failed validation: {err.args[0]}') from err



class Contract:
    def __init__(self, method, *, request_schema=None, response_schemas=None,
                 validate_request_data=True, validate_response_data=True,
                 additional_params=None):
        self.method = method.lower()
        self.request_schema = request_schema
        self.response_schemas = response_schemas
        self.validate_request_data = validate_request_data

        if validate_response_data:
            warnings.warn('validate_response_data is extremely ineffecient and will '
                          'result in returning confusing error messages to the user. '
                          'Use only during development for testing and debugging.')
        self.validate_response_data = validate_response_data
        self.additional_params = {} if additional_params is None else additional_params
        self._response_schema_lookup = {response_schema.status_code: response_schema
                                        for response_schema in response_schemas}

    def fetch_response_schema(self, status_code):
        return self._response_schema_lookup.get(status_code)
