"""
Use `robustify` to add documentation and schema validations to your existing
`flask` apps with minimal changes using open source standards like OpenAPI and
JSON Schema.
"""

import functools

import flask
import jsonschema

from porter.exceptions import InvalidModelInput

# Data Types


class ApiObject:
    """Simple abstractions providing an interface from `python` objects and
    popular API standards such as `openapi` and `jsonschema`.
    """
    def __init__(self, description=None, *, additional_params=None, reference_name=None):
        """
        Args:
            description (string): Description of the object.
            additional_params (None or dict): Key-Value pairs added to the
                objects OpenAPI definition.
            reference_name (None or str): If a `str` is given the object will
                be represented as a `$ref` in OpenAPI endpoint definitions and
                fully described by `reference_name` under `components/schemas`.
        """
        self.description = description
        self.additional_params = additional_params or {}
        self.reference_name = reference_name
        with _RefContext(ignore_refs=True):
            # On compatability with the OpenApi spec and json schema see
            # https://swagger.io/docs/specification/data-models/keywords/
            # and
            # http://json-schema.org/draft-06/json-schema-release-notes.html
            self.validator = jsonschema.Draft4Validator(self.to_openapi()[0])

    def to_openapi(self):
        """Return the OpenAPI definition of `self`.

        Returns:
            tuple: Returns two dicts, the first contains the OpenAPI
                definition of `self` and the second contains any references.
        """
        with _RefContext() as ref_context:
            openapi_spec = dict(type=self._openapi_type_name, description=self.description, **self.additional_params)
            openapi_spec.update(self._customized_openapi())
            if self.reference_name is not None and not _RefContext.context_ignore_refs():
                _RefContext.add_ref(self.reference_name, openapi_spec)
                return {'$ref': f'#/components/schemas/{self.reference_name}'}, ref_context.schemas
        return openapi_spec, ref_context.schemas

    def _customized_openapi(self):
        """Return a mapping of custom values to be added to the OpenAPI spec.
        Values specified here will override any defaults.
        """
        return {}

    @property
    def _openapi_type_name(self):
        return self.__class__.__name__.lower()


class String(ApiObject):
    """String type."""

class Number(ApiObject):
    """Number type."""

class Integer(ApiObject):
    """Integer type."""

class Boolean(ApiObject):
    """Boolean type."""


class Array(ApiObject):
    """Array type."""

    def __init__(self, *args, item_type=None,  **kwargs):
        """
        Args:
            *args: Positional arguments passed on to `ApiObject`.
            item_type (ApiObject): An ApiObject instance representing the item
                type stored in the array.
            **kwargs: Keyword arguments passed on to `ApiObject`.
        """
        self.item_type = item_type
        super().__init__(*args, **kwargs)

    def _customized_openapi(self):
        return {'items': self.item_type.to_openapi()[0]}

class Object(ApiObject):
    """Object type."""

    def __init__(self, *args, properties=None, required='all', **kwargs):
        """
        Args:
            *args: Positional arguments passed on to `ApiObject`.
            properties (dict): A mapping from property names to ApiObject
                instances.
            required ("all", `list`, `False): If "all" all properties are
                required, if a `list` only a subset are required. An empty
                list means all properties are optional.
            **kwargs: Keyword arguments passed on to `ApiObject`.
        """
        self.properties = properties
        if required == 'all':
            self.required = list(self.properties.keys())
        elif required and isinstance(required, list):
            self.required = required
        elif isinstance(required, list):
            self.required = []
        else:
            raise ValueError('required must be "all" or list')
        super().__init__(*args, **kwargs)

    def _customized_openapi(self):
        return {
            'properties': {name: prop.to_openapi()[0] for name, prop in self.properties.items()},
            'required': self.required
        }


class _RefContext:
    """Helper class to keep track of all referenced objects created when a
    nested data structure is converted its OpenAPI spec.

    Consider the following object

    ```yaml
    ObjectA:
      type: object
      properties:
        a:
          $ref: '#/components/schemas/ObjectB
    ```

    when `ObjectA` is converted to its OpenAPI spec we also need to return the
    definition of `ObjectB`. Additionally, `ObjectB` may contain a references
    to other objects itself that are needed to completely specify `ObjectA.

    We handle this by placing an instance of `_RefContext` on to a `stack`
    (last in/first out) every time an object is converted to its OpenAPI spec.
    Additionally each object "registers" the spec of any of its immediate
    references with `_RefContext` (which means they are attached to the first
    item in the stack). When the outer most object is ready to return all
    referenced dependencies will have attached their spec to the instance of
    `_RefContext` instantiated in that call.
    """

    _context = []

    def __init__(self, ignore_refs=False):
        self.schemas = {}
        self.ignore_refs = ignore_refs

    def __enter__(self):
        # it's tempting to do something like the following here:
        #     if not self._context:
        #         self._context.append(self)
        # but then we would never know when to clear _context in __exit__()
        self._context.append(self)
        return self

    def __exit__(self, *exc):
        self._context.pop()  # clean up the stack
        return False

    @classmethod
    def add_ref(cls, ref_name, openapi_spec):
        current_context = cls._context[0]  # always add definitions to the first
                                           # item in the stack!
        current_context.schemas[ref_name] = openapi_spec

    @classmethod
    def context_ignore_refs(cls):
        current_context = cls._context[0]  # always add definitions to the first
                                           # item in the stack!
        return current_context.ignore_refs


class RequestBody:
    def __init__(self, obj, description=None):
        self.obj = obj
        self.description = description

    def to_openapi(self):
        openapi_spec, openapi_refs = self.obj.to_openapi()
        content = {
            'application/json': {
                'schema': openapi_spec
            }
        }
        return {
            'requestBody': {
                'content': content,
                'description': self.description
            }
        }, openapi_refs


class ResponseBody:
    def __init__(self, description=None, *, status_code, obj):
        self.status_code = status_code
        self.obj = obj
        self.description = description

    def to_openapi(self):
        openapi_spec, openapi_refs = self.obj.to_openapi()
        content = {
            'application/json': {
                'schema': openapi_spec
            }
        }
        return {
            self.status_code: {
                'content': content,
                'description': self.description
            }
        }, openapi_refs


def attach_contract(method, *, request=None, responses=None, validate_request=False):
    """Decorator to attach API contract definitions to functions. Can be used to
    decorate `flask` routes.

    Args:
        method (str): "GET", "POST", etc.
        request (RequestBody): The schema for requests to the endpoint.
        responses (list[ResponseBody]): A list of responses from the endpoint.
        validate_request (bool): Validate the request data against the schema
            defined by `request`?

    Returns:
        function: The decorated function with the attribute `_contracts` assigned.

    Raises:
        jsonschema.exceptions.ValidationError
    """
    # set defaults
    request = None if request is None else request
    responses = [] if responses is None else responses

    def fn_decorator(fn):
        if not hasattr(fn, '_contracts'):
            fn._contracts = {}
        fn._contracts[method.lower()] = (request, responses)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if validate_request:
                method = flask.request.method.lower()
                method_spec = fn._contracts[method]

                if method_spec is not None and hasattr(method_spec[0], 'obj'):  # first item in tuple is requst
                    post_data = flask.request.get_json(force=True)
                    obj = method_spec[0].obj
                    _validate(post_data, obj)
            return fn(*args, **kwargs)

        return wrapper

    return fn_decorator


def _validate(data, obj):
    # TODO: register this as 422 on the flask app
    return obj.validator.validate(data)
