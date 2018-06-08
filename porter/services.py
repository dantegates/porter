"""Tools for building RESTful services that exposes machine learning models.

Building and running an app with the tools in this module is as simple as

1. Instantiating `ModelApp`.
2. Instantiating `ServiceConfig` once for each model you wish to add to the
    service.
3. Use the config(s) created in 2. to add models to the app with either
    `ModelApp.add_service()` or `ModelApp.add_services()`.

For example,

    >>> model_app = ModelApp()
    >>> service_config1 = ServiceConfig(...)
    >>> service_config2 = ServiceConfig(...)
    >>> model_app.add_services(service_config1, service_config2)
"""

import json

import flask
import numpy as np
import pandas as pd

import porter.responses as porter_responses


_ID_KEY = 'id'


class NumpyEncoder(json.JSONEncoder):
    """A JSON encoder that handles `numpy` data types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


class ServePrediction(object):
    """Class for building stateful prediction routes.

    Instances of this class are intended to be routed to endpoints in a `flask`
    app. E.g.

        >>> app = flask.Flask(__name__)
        >>> serve_prediction = ServePrediction(...)
        >>> app.route('/prediction/', methods=['POST'])(serve_prediction)

    Instances of this class can hold all required state necessary for making
    predictions at runtime and when called will return predictions corresponding
    POST requests sent to the app.
    """
    _instances = 0

    def __new__(cls, *args, **kwargs):
        # flask looks for the __name__ attribute of the routed callable,
        # and each name of a routed object must be unique.
        # Therefore we define a unique name here to meet flask's expectations.
        instance = super(ServePrediction, cls).__new__(cls)
        cls._instances += 1
        instance.__name__ = '%s_%s' % (cls.__name__.lower(), cls._instances)
        return instance

    def __init__(self, model, model_id, preprocessor, postprocessor, input_schema,
                 allow_nulls):
        """Initialize an instance of ServePrediction.

        Args:
            model (object): An object implementing the interface defined by
                `porter.datascience.BaseModel`.
            model_id (str): A unique identifier for the model. Returned to the
                user alongside the model predictions.
            preprocessor (object or None): An object implementing the interface
                defined by `porter.datascience.BaseProcessor`. If not `None`,
                the `.process()` method of this object will be called on the POST
                request data and its output will be passed to `model.predict()`.
            postprocessor (object or None): An object implementing the interface
                defined by `porter.datascience.BaseProcessor`. If not `None`,
                the `.process()` method of this object will be called on the
                output of `model.predict()` and its return value will be used to
                populate the predictions returned to the user. 
            input_schema (list-like or None): A list (or list like object) containing
                the feature names required in the POST data. Will be used to validate
                the POST request if not `None`.
            allow_nulls (bool): Are nulls allowed in the POST request data? If `False`
                an error is raised when nulls are found.
        """

        self.model = model
        self.model_id = model_id
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.input_schema = input_schema
        self.allow_nulls = allow_nulls
        self.validate_input = self.input_schema is not None
        self.preprocess_model_input = self.preprocessor is not None
        self.postprocess_model_output = self.postprocessor is not None

    def __call__(self):
        """Retrive POST request data from flask and return a response containing
        the corresponding predictions."""
        data = flask.request.get_json(force=True)
        X = pd.DataFrame(data)
        if self.validate_input:
            self.check_request(X, list(self.input_schema), self.allow_nulls)
        if self.preprocess_model_input:
            X = self.preprocessor.process(X)
        preds = self.model.predict(X)
        if self.postprocess_model_output:
            preds = self.postprocessor.process(preds)
        response = porter_responses.make_prediction_response(self.model_id, X[_ID_KEY], preds)
        return response

    @staticmethod
    def check_request(X, feature_names, allow_nulls=False):
        """Check the POST request data raising an error if a check fails.

        Checks include

        1. `X` contains all columns in `feature_names`.
        2. `X` does not contain nulls (only if allow_nulls == True).

        Args:
            X (pandas.DataFrame): A `pandas.DataFrame` created from the POST
                request.
            feature_names (list): All feature names expected in `X`.
            allow_nulls (bool): Whether nulls are allowed in `X`. False by
                default.

        Returns:
            None

        Raises:
            ValueError if a given check fails.
        """
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


def serve_root():
    """Return a helpful description of how to use the app."""
    message = (
        "I'm alive.<br>"
        'Send POST requests to /&lt model-name &gt/prediction/'
    )
    return message, 200


class PredictionServiceConfig:
    """
    A simple container that holds all necessary data for an instance of `ModelApp`
    to route a model.
    """
    def __init__(self, model, model_id, endpoint, preprocessor=None,
                 postprocessor=None, input_schema=None, allow_nulls=False):
        """
        Initialize a ServiceConfig.

        Args:
            model (object): An object implementing the interface defined by
                `porter.datascience.BaseModel`.
            model_id (str): A unique identifier for the model. Returned to the
                user alongside the model predictions.
            endpoint (str): Name of the model endpoint. The final routed endpoint
                will become "/<endpoint>/prediction/".
            preprocessor (object or None): An object implementing the interface
                defined by `porter.datascience.BaseProcessor`. If not `None`,
                the `.process()` method of this object will be called on the POST
                request data and its output will be passed to `model.predict()`.
                Optional.
            postprocessor (object or None): An object implementing the interface
                defined by `porter.datascience.BaseProcessor`. If not `None`,
                the `.process()` method of this object will be called on the
                output of `model.predict()` and its return value will be used to
                populate the predictions returned to the user. Optional.
            input_schema (list-like or None): A list (or list like object) containing
                the feature names required in the POST data. Will be used to validate
                the POST request if not `None`. Optional.
            allow_nulls (bool): Are nulls allowed in the POST request data? If `False`
                an error is raised when nulls are found. Optional.
        """
        self.model = model
        self.model_id = model_id
        self.endpoint = endpoint
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.input_schema = input_schema
        self.allow_nulls = allow_nulls


class ModelApp:
    """
    Abstraction used to simplify building REST APIs that expose predictive models.

    Essentially this class is a wrapper around an instance of `flask.Flask`.
    """

    _prediction_endpoint_template = '/{endpoint}/prediction/'
    _error_codes = (
        400,  # bad request
        404,  # not found
        405,  # method not allowed
        500,  # internal server error
    )

    def __init__(self):
        """Initialize an instance of `ModelApp`."""
        self.app = self._build_app()

    def add_services(self, *service_configs):
        for service_config in service_configs:
            self.add_service(service_config)

    def add_service(self, service_config):
        if isinstance(service_config, PredictionServiceConfig):
            self.add_prediction_service(service_config)
        else:
            raise ValueError('unkown service type')

    def add_prediction_service(self, service_config):
        """
        Add a model service to the API.

        Args:
            service_config (object): Instance of `porter.services.ServiceConfig`.
        Returns:
            None
        """
        prediction_endpoint = self._prediction_endpoint_template.format(
            endpoint=service_config.endpoint)
        serve_prediction = ServePrediction(
            model=service_config.model,
            model_id=service_config.model_id,
            preprocessor=service_config.preprocessor,
            postprocessor=service_config.postprocessor,
            input_schema=service_config.input_schema,
            allow_nulls=service_config.allow_nulls)
        self.app.route(prediction_endpoint, methods=['POST'])(serve_prediction)

    def run(self, *args, **kwargs):
        """
        Run the app.

        Args:
            *args: Positional arguments passed on to the wrapped `flask` app.
            **kwargs: Keyword arguments passed on to the wrapped `flask` app.
        """
        self.app.run(*args, **kwargs)

    def _build_app(self):
        """Build and return the `flask` app.

        Any global properties of the app, such as error handling and response
        formatting, are added here.
        """
        app = flask.Flask(__name__)
        # register a custom JSON encoder that handles numpy data types.
        app.json_encoder = NumpyEncoder
        # register error handlers
        for error in self._error_codes:
            app.register_error_handler(error, serve_error_message)
        # This route that can be used to check if the app is running.
        # Useful for kubernetes/helm integration
        app.route('/', methods=['GET'])(serve_root)
        return app
