import collections
import json

from .robustify import ResponseBody



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
    for endpoint, api_specs in _iter_contracts(app):
        for method, (request_spec, response_specs) in api_specs.items():
            paths[endpoint][method] = path_dict = {}
            path_dict['responses'] = {}

            if request_spec is not None:
                obj_spec, obj_refs = request_spec.to_openapi()
                path_dict.update(obj_spec)
                schemas.update(obj_refs)

            for response_spec in response_specs:
                obj_spec, obj_refs = response_spec.to_openapi()
                path_dict['responses'].update(obj_spec)
                schemas.update(obj_refs)
                
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
_swagger_html = '''
<!-- HTML for static distribution bundle build -->
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8">
    <title>Swagger UI</title>
    <link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@3/swagger-ui.css" >
    <link rel="icon" type="image/png" href="https://cdn.jsdelivr.net/npm/favicon-32x32.png" sizes="32x32" />
    <link rel="icon" type="image/png" href="https://cdn.jsdelivr.net/npm/favicon-16x16.png" sizes="16x16" />
    <style>
      html
      {
        box-sizing: border-box;
        overflow: -moz-scrollbars-vertical;
        overflow-y: scroll;
      }

      *,
      *:before,
      *:after
      {
        box-sizing: inherit;
      }

      body
      {
        margin:0;
        background: #fafafa;
      }
    </style>
  </head>

  <body>
    <div id="swagger-ui"></div>

    <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@3/swagger-ui-bundle.js"> </script>
    <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@3/swagger-ui-standalone-preset.js"> </script>
    <script>
    window.onload = function() {
      // Begin Swagger UI call region
      const ui = SwaggerUIBundle({
        url: "/docs.json",
        dom_id: '#swagger-ui',
        deepLinking: true,
        presets: [
          SwaggerUIBundle.presets.apis,
          SwaggerUIStandalonePreset
        ],
        plugins: [
          SwaggerUIBundle.plugins.DownloadUrl
        ],
        layout: "StandaloneLayout"
      })
      // End Swagger UI call region

      window.ui = ui
    }
  </script>
  </body>
</html>
'''
