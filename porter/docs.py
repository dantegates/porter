import collections
import json
import os

from .schemas import ResponseBody



def route_docs(app, title, description, api_version, html_endpoint, json_endpoint):
    openapi_json = _app_to_openapi(app, title, description, api_version)

    @app.route(json_endpoint)
    def docs_openapi():
        return json.dumps(openapi_json)

    @app.route(html_endpoint)
    def docs():
        return _swagger_html


def _app_to_openapi(app, title, description, api_version):
    openapi_spec = {
        'openapi': '3.0.1',
        'info': {
          'title': title,
          'description': description,
          'version': api_version
        },
        'paths': collections.defaultdict(lambda: collections.defaultdict(dict)),
        'components': {
            'schemas': {}
        }
    }
    paths = openapi_spec['paths']
    schemas = openapi_spec['components']['schemas']
    for endpoint, contracts in _iter_contracts(app):
        for method, contract in contracts.items():
            paths[endpoint][method] = path_dict = {}
            path_dict['responses'] = {}

            if contract.request_schema is not None:
                obj_spec, obj_refs = contract.request_schema.to_openapi()
                path_dict.update(obj_spec)
                schemas.update(obj_refs)

            if contract.response_schemas is not None:
                for response_schema in contract.response_schemas:
                    obj_spec, obj_refs = response_schema.to_openapi()
                    path_dict['responses'].update(obj_spec)
                    schemas.update(obj_refs)

            path_dict.update(contract.additional_params)
                
    return openapi_spec


def _iter_contracts(app):
    for rule in app.url_map.iter_rules():
        if rule.endpoint != 'static':
            rule_func = app.view_functions[rule.endpoint]
            if hasattr(rule_func, '_contracts'):
                yield rule.rule, rule_func._contracts


# in theory we could template this, but it's the only instance of returning
# html like this so why bother?
# https://github.com/swagger-api/swagger-ui/blob/master/dist/index.html
# TODO: parameterize where swagger scripts come from? include these as static?
with open(os.path.join(os.path.dirname(__file__), 'assets/swagger.html')) as f:
    _swagger_html = f.read()
