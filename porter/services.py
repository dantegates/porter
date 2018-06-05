import flask
import pandas as pd

import porter.responses as porter_responses
from porter import utils


_ID_KEY = 'id'


class ServePrediction(object):
    def __init__(self, model, model_id, feature_engineer, input_schema,
                 validate_input, allow_nulls):
        self.model = model
        self.model_id = model_id
        self.feature_engineer = feature_engineer
        self.input_schema = input_schema
        self.validate_input = validate_input
        self.allow_nulls = allow_nulls
        self.transform_input = self.feature_engineer is not None

    def serve(self):
        data = flask.request.get_json(force=True)
        X = pd.DataFrame(data)
        if self.validate_input:
            self.check_request(X, self.input_schema.keys(), self.allow_nulls)
        X_tf = self.feature_engineer.transform(X) if self.transform_input else X
        model_prediction = self.model.predict(X_tf)
        response = porter_responses.make_prediction_response(
            self.model_id, X[_ID_KEY], model_prediction)
        return response

    @staticmethod
    def check_request(X, feature_names, allow_nulls=False):
        required_keys = [_ID_KEY]
        required_keys.extend(feature_names)
        # checks that all columns are present and no nulls sent
        # (or missing values)
        try:
            # check for allow_nulls first to avoid computation if possible
            if not allow_nulls and X[required_keys].isnull().any().any():
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


def serve_error_message(error):
    """Return a response with JSON payload describing the most recent exception."""
    response = porter_responses.make_error_response(error)
    return response


def serve_alive():
    message = (
        "I'm alive.<br>"
        'Send POST requests to /&lt model-name &gt/prediction/'
    )
    return message, 200


class ServiceConfig:
    def __init__(self, model, model_id, endpoint, feature_engineer=None,
                 input_schema=None, validate_input=False, allow_nulls=False):
        self.model = model
        self.endpoint = endpoint
        self.model_id = model_id
        self.feature_engineer = feature_engineer
        self.input_schema = input_schema
        if validate_input and not self.input_schema:
            raise ValueError('input_schema is required when validate_input=True')
        self.validate_input = validate_input
        self.allow_nulls = allow_nulls


class ModelApp:
    _prediction_endpoint_template = '/{endpoint}/prediction/'
    _error_codes = (
        400,  # bad request
        404,  # not found
        405,  # method not allowed
        500   # internal server error
    )

    def __init__(self):
        self._build_app()

    def add_service(self, service_config):
        prediction_endpoint = self._prediction_endpoint_template.format(
            endpoint=service_config.endpoint)
        self._route_prediction(service_config, prediction_endpoint)

    def _route_prediction(self, service_config, prediction_endpoint):
        serve_prediction = ServePrediction(
            model=service_config.model,
            model_id=service_config.model_id,
            feature_engineer=service_config.feature_engineer,
            input_schema=service_config.input_schema,
            validate_input=service_config.validate_input,
            allow_nulls=service_config.allow_nulls)
        # flask looks for the __name__ attribute of the routed callable.
        # Hence we route a bound instance method rather than an instance
        # implementing __call__()
        self.app.route(prediction_endpoint, methods=['POST'])(serve_prediction.serve)

    def _build_app(self):
        self.app = flask.Flask(__name__)
        # register a custom JSON encoder that handles numpy data types.
        self.app.json_encoder = utils.NumpyEncoder
        # register error handlers
        for error in self._error_codes:
            self.app.register_error_handler(error, serve_error_message)
        # create a route that can be used to check if the app is running
        # useful for kubernetes/helm integration
        self._route_alive()

    def _route_alive(self):
        self.app.route('/', methods=['GET'])(serve_alive)
