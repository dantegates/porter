"""Tools for integrating the OpenAPI standard in ``porter``."""

import os

import fastjsonschema
from jinja2 import Template

from ..constants import ASSETS_DIR


def _numpy_to_builtin(x):
    """Convert NumPy dtypes (int32, float64, etc) to built-ins (int, float, etc)"""
    if hasattr(x, 'keys'):
        # mapping
        keys = list(x.keys())
    elif hasattr(x, '__len__'):
        # sequence
        keys = range(len(x))
    else:
        return
    for k in keys:
        if hasattr(x[k], 'tolist'):
            x[k] = x[k].tolist()
        elif isinstance(x[k], dict):
            _numpy_to_builtin(x[k])


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
                fully described by `reference_name` under "components/schemas".
        """
        self.description = description
        self.additional_params = additional_params or {}
        self.reference_name = reference_name
        with _RefContext(ignore_refs=True):
            # On compatability with the OpenApi spec and json schema see
            # https://swagger.io/docs/specification/data-models/keywords/
            # and
            # http://json-schema.org/draft-06/json-schema-release-notes.html
            self._jsonschema = self.to_openapi()[0]
            self._validate = fastjsonschema.compile({
                '$draft': '04',
                **self._jsonschema
            })

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

    def validate(self, data):
        """
        Args:
            data (JSON-like data structure): `data` will be evaulated against
                the OpenAPI spec that describes `self`. It should be a
                "JSON-like" data structure consisting of types compatible with
                `dict`, `list`, `int`, `str`, etc.

        Returns:
            None

        Raises:
            ValueError: If `data` does not conform to the OpenAPI spec that
                describes `self`. All errors messages will be prefixed with
                "Schema validation failed:" so that users can programatically
                differentiate between `ValueError`s explicitly raised by this
                method and others.

        """
        # possible hack for accepting numpy types
        #_numpy_to_builtin(data)
        try:
            self._validate(data)
        except fastjsonschema.exceptions.JsonSchemaException as err:
            # fastjsonschema raises useful error messsages so we'll reuse them.
            # However, a ValueError so that other modules don't need to depend
            # on fastjsonschema exceptions
            raise ValueError(f'Schema validation failed: {err.args[0]}', *err.args[1:]) from err


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

    def __init__(self, *args, item_type=None, **kwargs):
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

    def __init__(self, *args, properties=None, additional_properties_type=None, required='all', **kwargs):
        """
        Args:
            *args: Positional arguments passed on to `ApiObject`.
            properties (dict): A mapping from property names to ApiObject
                instances.
            additional_properties_type (:class:`ApiObject`): If this is a "free form" object,
                this defines the type of the additional properties.
            required ("all" or sequence): If "all", all properties are required; if a
                sequence, only a subset are required. An empty list means all properties are
                optional.
            **kwargs: Keyword arguments passed on to :class:`ApiObject`.

        Raises:
            ValueError: If both ``properties`` and ``additional_properties_type`` are None.
        """
        if properties is None and additional_properties_type is None:
            raise ValueError('at least one of properties and additional_properties_type should be specified')
        self.properties = properties
        self.additional_properties_type = additional_properties_type
        if properties is not None:
            if required == 'all':
                self.required = tuple(sorted(self.properties.keys()))
            else:
                self.required = tuple(required)
        super().__init__(*args, **kwargs)

    def _customized_openapi(self):
        spec = {}
        if self.properties is not None:
            spec['properties'] = {name: prop.to_openapi()[0] for name, prop in self.properties.items()}
            spec['required'] = self.required
        if self.additional_properties_type is not None:
            if hasattr(self.additional_properties_type, 'to_openapi'):
                spec['additionalProperties'] = self.additional_properties_type.to_openapi()[0]
            else:
                spec['additionalProperties'] = self.additional_properties_type
        return spec


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


class RequestSchema:
    def __init__(self, api_obj, description=None):
        self.api_obj = api_obj
        self.description = description

    def to_openapi(self):
        openapi_spec, openapi_refs = self.api_obj.to_openapi()
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


class ResponseSchema:
    def __init__(self, api_obj, status_code, description=None):
        self.status_code = status_code
        self.api_obj = api_obj
        self.description = description

    def to_openapi(self):
        openapi_spec, openapi_refs = self.api_obj.to_openapi()
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


def make_openapi_spec(title, description, version, request_schemas, response_schemas,
                      additional_params):
    """
    Args:
        title (str): The title of the application.
        description (str): A description of the application.
        version (str): The version of the application.
        request_schemas (dict): Nested dictionary mapping endpoints to a
            dictionary of HTTP methods to instances of :class:`RequestSchema`.
            E.g. `{"/foo/bar": {"GET": RequestSchema(...)}}`.
        response_schemas (dict): Nested dictionary mapping endpoints to
            a dictionary of HTTP methods to lists of instances of
            :class:`ResponseSchema`.
            E.g. `{"/foo/bar/": {"GET": [ResponseSchema(...), ResponseSchema(...)]}}`
        additional_params (dict): A nested dictionary mapping tuples of
            endpoints and HTTP methods to a dictionary containing arbitrary
            OpenAPI values that will be applied to the OpenAPI spec for that
            endpoint/method.
            E.g. `{("/foo/bar/", 'GET): {"tags": ["tag1", "tag2"]}}`

    Returns:
        dict: The OpenAPI spec describing the provided arguments.
    """
    paths = _init_paths(request_schemas, response_schemas, additional_params)
    components_schemas = {}
    spec = {
        'openapi': '3.0.1',
        'info': {
            'title': title,
            'description': description,
            'version': version
        },
        'paths': paths,
        'components': {
            'schemas': components_schemas
        }
    }

    for endpoint, requests in request_schemas.items():
        for method, schema in requests.items():
            paths[endpoint][method.lower()] = method_dict = {}
            _update_spec(schema, method_dict, components_schemas)

    for endpoint, responses in response_schemas.items():
        for method, schemas in responses.items():
            for schema in schemas:
                if not 'responses' in paths[endpoint][method.lower()]:
                    paths[endpoint][method.lower()]['responses'] = {}
                method_dict = paths[endpoint][method.lower()]['responses']
                _update_spec(schema, method_dict, components_schemas)

    for endpoint, params in additional_params.items():
        for method, params in params.items():
            method_dict = paths[endpoint][method.lower()]
            method_dict.update(params)

    return spec


def _init_paths(request_schemas, response_schemas, additional_params):
    endpoints = (
        set(request_schemas.keys())
        | set(response_schemas.keys())
        | set(additional_params.keys())
    )
    endpoint_methods = {
        endpoint: (set(request_schemas.get(endpoint, {}).keys())
                 | set(response_schemas.get(endpoint, {}).keys())
                 | set(additional_params.get(endpoint, {}).keys()))
        for endpoint in endpoints
    }
    paths = {}
    return {endpoint: {method.lower(): {} for method in methods}
            for endpoint, methods in endpoint_methods.items()}


def _update_spec(schema, method_dict, components_schemas):
    spec, refs = schema.to_openapi()
    method_dict.update(spec)
    components_schemas.update(refs)


# https://github.com/swagger-api/swagger-ui/blob/master/dist/index.html
with open(os.path.join(ASSETS_DIR, 'swagger-ui/swagger_template.html')) as f:
    _doc_template = Template(f.read())


def make_docs_html(docs_prefix, docs_json_url):
    """
    Args:
        docs_json_url (str): URL where documentation JSON is exposed. Ignored if
            ``expose_docs=False``.

    Returns:
        str: Static html docs to serve.
    """
    return _doc_template.render(docs_prefix=docs_prefix, docs_json_url=docs_json_url)
