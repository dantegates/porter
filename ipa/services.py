from functools import partial

import flask
import pandas as pd
from werkzeug.exceptions import BadRequest

from ipa.responses import make_prediction_response, make_error_response
from ipa import utils


_ID_KEY = 'id'


def check_request(X, feature_names):
    required_keys = [_ID_KEY]
    required_keys.extend(feature_names)
    # checks that all columns are present and no nulls sent
    # (or missing values)
    try:
        if X[required_keys].isnull().any().any():
            null_counts = X[required_keys].isnull().sum()
            null_columns = null_counts[null_counts > 0].index.tolist()
            raise ValueError(
                'request payload had null values in the following fields: %s'
                % null_columns)
    except KeyError:
        missing = [c for c in required_keys if not c in X.columns]
        raise ValueError(
            'request payload is missing the following fields: %s'
            % missing)

def serve_prediction(model, feature_engineer, input_schema, validate_request):
    data = flask.request.get_json(force=True)
    X = pd.DataFrame(data)
    if validate_request:
        try:
            validate_request(X, input_schema.keys())
        except ValueError:
            raise BadRequest()
    X_tf = X if feature_engineer is None else feature_engineer.transform(X)
    model_prediction = model.predict(X_tf)
    response = make_prediction_response(model.id, X[_ID_KEY], model_prediction)
    return response

def serve_error_message(error):
    """Return a response with JSON payload describing the most recent exception."""
    response = make_error_response(error)
    return response


class ModelServiceConfig:
    def __init__(self, model, feature_engineer=None, input_schema=None,
                 validate_request=False):
        self.model = model
        self.feature_engineer = feature_engineer
        self.input_schema = input_schema
        if validate_request and not self.input_schema:
            raise ValueError('input_schema is required when validate_request=True')
        self.validate_request = validate_request


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

    def add_model_service(self, service_config):
        model = service_config.model
        model_name = service_config.model.name
        feature_engineer = service_config.feature_engineer
        input_schema = service_config.input_schema
        validate_request = service_config.validate_request
        model_url = self._make_model_url(model_name)
        fn = self._init_prediction_fn(model=model, feature_engineer=feature_engineer,
            input_schema=input_schema, validate_request=validate_request)
        self.app.route(model_url, methods=['POST'])(fn)

    def _build_app(self):
        app = flask.Flask(__name__)
        app.json_encoder = utils.NumpyEncoder
        for error in self._error_codes:
            app.register_error_handler(error, serve_error_message)
        return app

    def _make_model_url(self, model_name):
        return self._url_prediction_format.format(model_name=model_name)

    def _init_prediction_fn(self, model, feature_engineer=None, input_schema=None,
                            validate_request=False):
        fn = partial(serve_prediction, model=model,
            feature_engineer=feature_engineer, input_schema=input_schema,
            validate_request=validate_request)
        # mimic function API - assumed in flask implementation
        fn.__name__ = '{}_prediction'.format(model.name.replace('-', '_'))
        return fn
