import functools
import json
import warnings

import fastjsonschema

from ..api import request_json, request_method
from ..exceptions import InvalidModelInput


def attach_contracts(contracts):
    """Decorator to attach API contract definitions to functions. Can be used to
    decorate `flask` routes.

    Args:
        contracts (`list[Contract]`): The contract for the endpoint.
        validate_request (bool): Validate the request data against the schema
            defined by `request`?

    Returns:
        function: The decorated function with the attribute `_contracts` assigned.

    Raises:
        jsonschema.exceptions.ValidationError
    """

    def fn_decorator(fn):

        fn._contracts = {c.method: c for c in contracts}

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            method = request_method().lower()
            # TODO: raise error if validate request is True but the contract
            #       can't be found or warning?
            method_contract = fn._contracts.get(method)
            if method_contract is not None and method_contract.validate_request_data \
                    and method_contract.request_schema is not None:
                data = request_json()
                _validate_request(method_contract, data)

            response = fn(*args, **kwargs)

            if method_contract is not None and method_contract.validate_response_data \
                    and method_contract.request_schema is not None:
                _validate_response(method_contract, response)

            return response

        return wrapper

    return fn_decorator


def _validate_request(method_contract, data):
    try:
        method_contract.request_schema.obj.validate(data)
    except fastjsonschema.exceptions.JsonSchemaException as err:
        # fastjsonschema raises useful error messsages so we'll reuse them.
        # However, we'll raise a InvalidModelInput to signal that this is a
        # model context error.
        raise InvalidModelInput(f'{method_contract.method.upper()} data failed validation: {err.args[0]}') from err


def _validate_response(method_contract, response):
    try:
        response_schema = method_contract.fetch_response_schema(response.status_code)
        response_schema.obj.validate(json.loads(response.data))
    except fastjsonschema.exceptions.JsonSchemaException as err:
        # fastjsonschema raises useful error messsages so we'll reuse them.
        # However, we'll raise a buitin ValueError so we don't need to depend
        # on fastjsonschema objects in other modules.
        raise Exception(f'{method_contract.method.upper()} return data failed validation: {err.args[0]}') from err



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
