import functools

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
                try:
                    method_contract.request_schema.obj.validate(data)
                except fastjsonschema.exceptions.JsonSchemaException as err:
                    # fastjsonschema raises useful error messsages so we'll reuse them.
                    # However, we'll raise a buitin ValueError so we don't need to depend
                    # on fastjsonschema objects in other modules.
                    raise InvalidModelInput(f'POST data failed validation: {err.args[0]}') from err

            return fn(*args, **kwargs)

        return wrapper

    return fn_decorator
