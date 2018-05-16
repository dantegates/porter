import traceback
from functools import partial

import flask
import pandas as pd
from werkzeug.exceptions import BadRequest


# TODO: model version in response
# TODO: handle id


def check_request(X, feature_names):
    # checks that all columns are present and no nulls sent
    # (or missing values)
    if X[[feature_names]].isnull().any().any():
        raise ValueError('No!')


def serve_prediction(model, feature_engineer=None):
    data = flask.request.get_json(force=True)
    X = pd.DataFrame(data)
    try:
        check_request(data, model.get_feature_names())
    except ValueError:
        raise BadRequest()
    if feature_engineer is not None:
        X = feature_engineer.transform(X)
    model_prediction = model.predict(X)
    return flask.jsonify(model_prediction)


def serve_schema(model_schema):
    return flask.jsonify(model_schema)


def serve_error_message(error):
    """Return a response with JSON payload describing the most recent exception."""
    tb = traceback.format_exc()
    error_body = {
        'error': type(error).__name__,
        # getattr() is used to work around werkzeug's bad implementation
        # of HTTPException (i.e. HTTPException inherits from Exception but
        # exposes a different API, namely
        # Exception.message -> HTTPException.description).
        'message': getattr(error, 'description', error.message),
        'traceback': tb}
    response = flask.jsonify(error_body)
    response.status_code = getattr(error, 'code', 500)
    return response


class ModelService:
    _url_prediction_format = '{model_name}/prediction/'
    _url_schema_format = '{model_name}/schema/'
    _error_codes = (
        400,  # bad request
        404,  # not found
        405,  # method not allowed
        500   # internal server error
    )

    def __init__(self):
        self._app = self._build_app()

    def run(self, **kwargs):
        self._app.run(**kwargs)

    def add_model(self, model, name, feature_engineer=None):
        self.route_model(model, name, feature_engineer=feature_engineer)
        self.route_model_schema(model.get_schema(), name)

    def route_model(self, model, name, feature_engineer=None):
        model_url = self._make_model_url(name)
        fn = self._make_model_prediction_fn(model, feature_engineer)
        self._app.route(model_url, methods=['POST'])(fn)

    def route_model_schema(self, model_schema, name):
        schema_url = self._make_model_schema_url(name)
        fn = self._make_model_schema_fn(model_schema)
        self._app.route(schema_url, methods=['GET'])(fn)

    def _build_app(self):
        app = flask.Flask(__name__)
        self._add_exception_handlers(app, self._error_codes)

    def _make_model_url(self, name):
        return self._url_prediction_format.format(model_name=name)

    def _make_model_schema_url(self, name):
        return self._url_schema_format.format(model_name=name)

    def _make_model_prediction_fn(self, model, feature_engineer=None):
        return partial(serve_prediction, model=model, feature_engineer=feature_engineer)

    def _make_model_schema_fn(self, model_schema):
        return partial(serve_schema, model_schema=model_schema)

    def _add_exception_handlers(self, app, error_codes):
        """Register a generic function to handle given error codes."""
        for error in error_codes:
            app.register_error_handler(error, serve_error_message)
