import traceback
from functools import partial

import flask
import pandas as pd
from werkzeug.exceptions import BadRequest

import responses
import utils


_ID_KEY = 'id'


def check_request(X, feature_names):
    required_keys = [_ID_KEY]
    required_keys.extend(feature_names)
    # checks that all columns are present and no nulls sent
    # (or missing values)
    if X[required_keys].isnull().any().any():
        raise ValueError('No!')


def serve_prediction(model, feature_engineer):
    data = flask.request.get_json(force=True)
    X = pd.DataFrame(data)
    try:
        check_request(X, feature_engineer.get_feature_names())
    except ValueError:
        raise BadRequest()
    X_tf = feature_engineer.transform(X)
    model_prediction = model.predict(X_tf)
    resp = responses.PredictionResponse(model.id, X[_ID_KEY], model_prediction)
    return flask.jsonify(resp)


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
    _url_prediction_format = '/{model_name}/prediction/'
    _error_codes = (
        400,  # bad request
        404,  # not found
        405,  # method not allowed
        500   # internal server error
    )

    def __init__(self):
        self.app = self._build_app()

    def add_model(self, model, feature_engineer):
        model_url = self._make_model_url(model.name)
        fn = self._make_model_prediction_fn(model, feature_engineer)
        self.app.route(model_url, methods=['POST'])(fn)

    def _build_app(self):
        app = flask.Flask(__name__)
        app.json_encoder = utils.NumpyEncoder
        for error in self._error_codes:
            app.register_error_handler(error, serve_error_message)
        return app

    def _make_model_url(self, model_name):
        return self._url_prediction_format.format(model_name=model_name)

    def _make_model_prediction_fn(self, model, feature_engineer=None):
        partial_fn = partial(serve_prediction, model=model, feature_engineer=feature_engineer)
        # mimic function API
        partial_fn.__name__ = '{}_prediction'.format(model.name.replace('-', '_'))
        return partial_fn
