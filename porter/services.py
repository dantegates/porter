"""Tools for building RESTful services that exposes machine learning models.

Building and running an app with the tools in this module is as simple as

1. Instantiating `ModelApp`.
2. Instantiating a "service". E.g. instantiate `PredictionService` for each
   model you wish to add to the service.
3. Use the service(s) created in 2. to add models to the app with either
    `ModelApp.add_service()` or `ModelApp.add_services()`.

For example,

    >>> model_app = ModelApp()
    >>> prediction_service1 = PredictionService(...)
    >>> prediction_service2 = PredictionService(...)
    >>> model_app.add_services(prediction_servie1, prediction_service2)

Now the model app can be run with `model_app.run()` for development, or as an
example of running the app in production `$ gunicorn my_module:model_app`.
"""

import abc
import json
import logging
import string
import warnings

import pandas as pd
import werkzeug.exceptions

from . import api
from . import config as cf
from . import constants as cn
from . import exceptions as exc
from . import responses as porter_responses

# alias for convenience
_ID = cn.PREDICTION_PREDICTIONS_KEYS.ID

_logger = logging.getLogger(__name__)


def serve_root():
    """Return a helpful description of how to use the app."""

    message = (
        'Send POST requests to /&lt model-name &gt/prediction/'
    )
    return message, 200


class StatefulRoute:
    """Helper class to ensure that classes we intend to route via their
    __call__() method satisfy the flask interface.
    """
    def __new__(cls, *args, **kwargs):
        # flask looks for the __name__ attribute of the routed callable,
        # and each name of a routed object must be unique.
        # Therefore we define a unique name here to meet flask's expectations.
        instance = super().__new__(cls)
        if not hasattr(cls, '_instances'):
            cls._instances = 0
        cls._instances += 1
        instance.__name__ = '%s_%s' % (cls.__name__.lower(), cls._instances)
        return instance


def serve_error_message(error):
    response = porter_responses.make_error_response(error)
    _logger.exception(response.data)
    return response.jsonify()


class ServeAlive(StatefulRoute):
    """Class for building stateful liveness routes.

    Args:
        app (object): A `ModelApp` instance. Instances of this class inspect
            `app` when called to determine if the app is alive.
    """

    logger = logging.getLogger(__name__)

    def __init__(self, app):
        self.app = app

    def __call__(self):
        """Serve liveness response."""
        response = porter_responses.make_alive_response(self.app)
        self.logger.info(response.data)
        return response.jsonify()


class ServeReady(StatefulRoute):
    """Class for building stateful readiness routes.

    Args:
        app (object): A `ModelApp` instance. Instances of this class inspect
            `app` when called to determine if the app is alive.
    """

    logger = logging.getLogger(__name__)

    def __init__(self, app):
        self.app = app

    def __call__(self):
        """Serve readiness response."""
        response = porter_responses.make_ready_response(self.app)
        self.logger.info(response.data)
        return response.jsonify()


class PredictSchema:
    """
    A simple container that represents a model's schema.

    Args:
        input_features (list of str): A list of the features input to the
            model service. If the service defines a preprocessor these are the
            features expected by the preprocessor.

    Attributes:
        input_columns (list of str): A list of all columns expected in the
            POST request payload.
        input_features (list of str): A list of the features input to the
            model service. If the service defines a preprocessor these are the
            features expected by the preprocessor.
    """
    def __init__(self, *, input_features):
        if input_features is not None:
            self.input_columns = [_ID] + input_features
        else:
            self.input_columns = input_features
        self.input_features = input_features


class BaseService(abc.ABC, StatefulRoute):
    """
    Abstract base class for services.

    A service class contains all necessary state and functionality to route a
    service and serve requests.

    Args:
        name (str): The model name. The final routed endpoint is generally
            derived from this parameter.
        api_version (str): The service API version. The final routed endpoint is
            generally derived from this parameter.
        meta (dict): Additional meta data added to the response body. Optional.
        log_api_calls (bool): Log request and response and response data.
            Default is False.
        namespace (str): String identifying a namespace that the service belongs
            to. Used to route services by subclasses. Default is "".

    Attributes:
        id (str): A unique ID for the service.
        name (str): The model name. The final routed endpoint is generally
            derived from this attribute.
        api_version (str): The service version.
        meta (dict): Additional meta data added to the response body. Optional.
        log_api_calls (bool): Log request and response and response data.
            Default is False.
        namespace (str): A namespace that the service belongs to.
        endpoint (str): The endpoint where the service is exposed.
        action (str): `str` describing the action of the service, e.g.
            "prediction". Used to determine the final routed endpoint.
    """
    _ids = set()
    _logger = logging.getLogger(__name__)

    def __init__(self, *, name, api_version, meta=None, log_api_calls=False, namespace=''):
        self.name = name
        self.api_version = api_version
        self.meta = {} if meta is None else meta
        self.check_meta(self.meta)
        self.namespace = namespace
        # Assign endpoint and ID last so they can be determined from other
        # instance attributes. If the order of assignment changes here these
        # methods may attempt to access attributes that have not been set yet
        # and fail.
        self.endpoint = self.define_endpoint()
        self.id = self.define_id()
        self.meta = self.update_meta(self.meta)
        self.log_api_calls = log_api_calls

    def __call__(self):
        """Serve a response to the user."""
        response = None
        try:
            response = self.serve()
            if not isinstance(response, porter_responses.Response):
                response = porter_responses.Response(response, service_class=self)
            response = response.jsonify()
        except exc.ModelContextError as err:
            err.update_model_context(self)
            self._log_error(err)
            raise err
        # technically we should never get here. self.serve() should always
        # return a ModelContextError but I'm a little paranoid about this.
        except Exception as err:
            self._log_error(err)
            raise err
        finally:
            if self.log_api_calls:
                request_data = api.request_json()
                if response is not None:
                    response_data = getattr(response, 'raw_data', response)
                else:
                    response_data = response
                self._log_api_call(request_data, response_data)
        return response

    def define_endpoint(self):
        """Return the service endpoint derived from instance attributes."""
        endpoint = cn.ENDPOINT_TEMPLATE.format(
            namespace=self.namespace, service_name=self.name,
            api_version=self.api_version, action=self.action)
        return endpoint

    @abc.abstractmethod
    def serve(self):
        """Return a response to be served to the user (usually the return
        value of one of the functions in `porter.responses` or an instance of
        `porter.responses.Response`).

        Custom subclasses may find it easier to return a native Python object
        such as a `str` or `dict`, in such cases the object must be
        "jsonify-able".
        """

    @abc.abstractproperty
    def status(self):
        """Return `str` representing the status of the service."""

    @property
    def route_kwargs(self):
        """Keyword arguments to use when routing `self.serve()`."""
        return {}

    @property
    @abc.abstractproperty
    def action(self):
        """`str` describing the action of the service, e.g. "prediction".
        Used to determine the final routed endpoint.
        """

    def define_id(self):
        """Return a unique ID for the service. This is used to set the `id`
        attribute.
        """
        return self.endpoint

    def check_meta(self, meta):
        """Raise `ValueError` if `meta` contains invalid values, e.g. `meta`
        cannot be converted to JSON properly.

        Subclasses overriding this method should always use super() to call
        this method on the superclass unless they have a good reason not to.
        """
        try:
            # sort_keys=True tests the flask.jsonify implementation
            _ = json.dumps(meta, cls=cf.json_encoder, sort_keys=True)
        except TypeError:
            raise exc.PorterError(
                'Could not jsonify meta data. Make sure that meta data is '
                'valid JSON and that all keys are of the same type.')

    def update_meta(self, meta):
        """Update meta data with instance state if desired and return."""
        return meta

    @property
    def id(self):
        """A unique ID for the instance."""
        return self._id

    @id.setter
    def id(self, value):
        if value in self._ids:
            raise exc.PorterError(
                f'The id={value} has already been used. '
                'This likely means that you tried to instantiate a service '
                'with parameters that were already used.')
        self._ids.add(value)
        self._id = value

    @property
    def namespace(self):
        return self._namespace

    @namespace.setter
    def namespace(self, value):
        if value and not value.startswith('/'):
            value = '/' + value
        self._namespace = value

    @property
    def name(self):
        """The model name. The final routed endpoint is generally derived from
        this parameter.
        """
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def api_version(self):
        """The model version. The final routed endpoint is generally derived from
        this parameter.
        """
        return self._api_version

    @api_version.setter
    def api_version(self, value):
        self._api_version = value

    def get_post_data(self):
        return api.request_json(force=True)

    def _log_api_call(self, request_data, response_data):
        self._logger.info('api logging',
            extra={'request_id': api.request_id(),
                   'request_data': request_data,
                   'response_data': response_data,
                   'service_class': self.__class__.__name__,
                   'event': 'api_call'})

    def _log_error(self, error):
        self._logger.exception(error,
            extra={'request_id': api.request_id(),
                   'service_class': self.__class__.__name__,
                   'event': 'exception'})


class PredictionService(BaseService):
    """
    A prediction service. Instances can be added to instances of `ModelApp`
    to serve predictions.

    Args:
        name (str): The model name. The final routed endpoint will become
            "/<namespace>/<name>/<api version>/prediction/".
        api_version (str): The model API version. The final routed endpoint
            will become "/<namespace>/<name>/<api version>/prediction/".
        meta (dict): Additional meta data added to the response body. Optional.
        log_api_calls (bool): Log request and response and response data.
            Default is False.
        namespace (str): String identifying a namespace that the service belongs
            to. The final routed endpoint will become
            "/<namespace>/<name>/<api version>/prediction/". Default is "".
        model (object): An object implementing the interface defined by
            `porter.datascience.BaseModel`.
        preprocessor (object or None): An object implementing the interface
            defined by `porter.datascience.BaseProcessor`. If not `None`, the
            `.process()` method of this object will be called on the POST
            request data and its output will be passed to `model.predict()`.
            Optional.
        postprocessor (object or None): An object implementing the interface
            defined by `porter.datascience.BaseProcessor`. If not `None`, the
            `.process()` method of this object will be called on the output of
            `model.predict()` and its return value will be used to populate
            the predictions returned to the user. Optional.
        input_features (list-like or None): A list (or list like object)
            containing the feature names required in the POST data. Will be
            used to validate the POST request if not `None`. Optional.            
        allow_nulls (bool): Are nulls allowed in the POST request data? If
            `False` an error is raised when nulls are found. Optional.
        batch_prediction (bool): Whether or not batch predictions are
            supported or not. If `True` the API will accept an array of objects
            to predict on. If `False` the API will only accept a single object
            per request. Optional.
        additional_checks (callable): Raises `InvalidModelInput` or subclass thereof
            if POST request is invalid.

    Attributes:
        id (str): A unique ID for the model. Composed of `name` and `api_version`.
        name (str): The model's name.
        meta (dict): Additional meta data added to the response body. Optional.
        log_api_calls (bool): Log request and response and response data.
            Default is False.
        namespace (str): String identifying a namespace that the service belongs
            to. The final routed endpoint will become
            "/<namespace>/<name>/<api version>/prediction/". Default is "".
        api_version (str): The model API version.
        endpoint (str): The endpoint where the model predictions are exposed.
            This is computed as "/<name>/<api version>/prediction/".
        model (object): An object implementing the interface defined by
            `porter.datascience.BaseModel`.
        preprocessor (object or None): An object implementing the interface
            defined by `porter.datascience.BaseProcessor`. If not `None`, the
            `.process()` method of this object will be called on the POST
            request data and its output will be passed to `model.predict()`.
            Optional.
        postprocessor (object or None): An object implementing the interface
            defined by `porter.datascience.BaseProcessor`. If not `None`, the
            `.process()` method of this object will be called on the output of
            `model.predict()` and its return value will be used to populate
            the predictions returned to the user. Optional.
        schema (object): An instance of `porter.services.Schema`.
        allow_nulls (bool): Are nulls allowed in the POST request data? If
            `False` an error is raised when nulls are found. Optional.
        batch_prediction (bool): Whether or not the endpoint supports batch
            predictions or not. If `True` the API will accept an array of
            objects to predict on. If `False` the API will only accept a
            single object per request. Optional.
        additional_checks (callable): Raises `InvalidModelInput` or subclass thereof
            if POST request is invalid.
    """

    route_kwargs = {'methods': ['GET', 'POST'], 'strict_slashes': False}
    action = 'prediction'

    def __init__(self, *, model, preprocessor=None, postprocessor=None,
                 input_features=None, allow_nulls=False, action=None,
                 batch_prediction=False, additional_checks=None, **kwargs):
        self.model = model
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.schema = PredictSchema(input_features=input_features)
        self.allow_nulls = allow_nulls
        self.batch_prediction = batch_prediction
        if additional_checks is not None and not callable(additional_checks):
            raise exc.PorterError('`additional_checks` must be callable')
        self.action = action or self.action
        self.additional_checks = additional_checks
        self._validate_input = self.schema.input_columns is not None
        self._preprocess_model_input = self.preprocessor is not None
        self._postprocess_model_output = self.postprocessor is not None
        super().__init__(**kwargs)

    @property
    def status(self):
        """Return 'READY'. Instances of this class are always ready."""
        return cn.HEALTH_CHECK_VALUES.IS_READY

    def serve(self):
        """Retrive POST request data from flask and return a response
        containing the corresponding predictions.

        Returns:
            object: A "jsonified" object representing the response to return
                to the user.

        Raises:
            porter.exceptions.ModelContextError: Raised whenever an error
                occurs during prediction. The error contains information
                about the model context which a custom error handler can
                use to add to the errors response.
        """
        if api.request_method() == 'GET':
            return porter_responses.Response(
                'This endpoint is live. Send POST requests for predictions')
        try:
            response = self._predict()
        # All we have to do with the exception handling here is
        # signal to __call__() that this is a model context error.
        # If it's a werkzeug exception let it fall through.
        except (exc.ModelContextError, werkzeug.exceptions.HTTPException) as err:
            raise err
        # All other errors should be wrapped in a PredictionError and raise 500
        except Exception as err:
            error = exc.PredictionError('an error occurred during prediction')
            raise error from err
        return response

    def _predict(self):
        X_input = self.get_post_data()

        if self._validate_input:
            self.check_request(X_input, self.schema.input_columns,
                self.allow_nulls, self.additional_checks)
            X_preprocessed = X_input.loc[:,self.schema.input_features]
        else:
            X_preprocessed = X_input

        if self._preprocess_model_input:
            X_preprocessed = self.preprocessor.process(X_preprocessed)

        preds = self.model.predict(X_preprocessed)

        if self._postprocess_model_output:
            preds = self.postprocessor.process(X_input, X_preprocessed, preds)

        if self.batch_prediction:
            response = porter_responses.make_batch_prediction_response(
                self, X_input[_ID], preds)
        else:
            response = porter_responses.make_prediction_response(
                self, X_input[_ID].iloc[0], preds[0])

        return response

    @classmethod
    def check_request(cls, X_input, input_columns, allow_nulls=False, additional_checks=None):
        """Check the POST request data raising an error if a check fails.

        Checks include

        1. `X` contains all columns in `feature_names`.
        2. `X` does not contain nulls (only if allow_nulls == True).

        Args:
            X (`pandas.DataFrame`): A `pandas.DataFrame` created from the POST
                request.
            feature_names (list): All feature names expected in `X`.
            allow_nulls (bool): Whether nulls are allowed in `X`. False by
                default.

        Returns:
            None

        Raises:
            porter.exceptions.RequestContainsNulls: If the input contains nulls
                and `allow_nulls` is False.
            porter.exceptions.RequestMissingFields: If the input is missing
                required fields.
            porter.InvalidModelInput: If user defined `additional_checks` fails.
        """
        cls._default_checks(X_input, input_columns, allow_nulls)
        # Only perform user checks after the standard checks have been passed.
        # This allows the user to assume that all columns are present and there
        # are no nulls present (if allow_nulls is False).
        if additional_checks is not None:
            additional_checks(X_input)

    @staticmethod
    def _default_checks(X, input_columns, allow_nulls):
        # checks that all columns are present and no nulls sent
        # (or missing values)
        try:
            # check for allow_nulls first to avoid computation if possible
            if not allow_nulls and X[input_columns].isnull().any().any():
                null_counts = X[input_columns].isnull().sum()
                null_columns = null_counts[null_counts > 0].index.tolist()
                raise exc.RequestContainsNulls(null_columns)
        except KeyError:
            missing_fields = [c for c in input_columns if not c in X.columns]
            raise exc.RequestMissingFields(missing_fields)

    def get_post_data(self):
        """Return data from the most recent POST request as a `pandas.DataFrame`.

        Returns:
            `pandas.DataFrame`. Each `row` represents a single instance to
            predict on. If `self.batch_prediction` is `False` the `DataFrame`
            will only contain one `row`.

        Raises:
            porter.exceptions.PorterError: If the request data does not
                follow the API format.
        """
        data = super().get_post_data()
        if not self.batch_prediction:
            # if API is not supporting batch prediction user's must send
            # a single JSON object.
            if not isinstance(data, dict):
                raise exc.InvalidModelInput(f'input must be a single JSON object')
            # wrap the `dict` in a list to convert to a `DataFrame`
            data = [data]
        elif not isinstance(data, list):
            raise exc.InvalidModelInput(f'input must be an array of objects')
        return pd.DataFrame(data)
 

class PredictionServiceConfig(PredictionService):
    def __init__(self, *args, **kwargs):
        warnings.warn('PredictionServiceConfig is deprecated. Use PredictionService.')
        super().__init__(*args, **kwargs)


class ModelApp:
    """
    Abstraction used to simplify building REST APIs that expose predictive
    models.

    Essentially this class is a wrapper around an instance of `flask.Flask`.

    Args:
        meta (dict): Additional meta data added to the response body. Optional.
    """

    def __init__(self, meta=None, description=None):
        self.meta = {} if meta is None else meta
        self.check_meta(self.meta)

        self._services = []
        # this is just a cache of service IDs we can use to verify that
        # each service is given a unique ID
        self._service_ids = set()
        self.app = self._build_app()

    def __call__(self, *args, **kwargs):
        """Return a WSGI interface to the model app."""
        return self.app(*args, **kwargs)

    def add_services(self, *services):
        """Add services to the app from `*services`.

        Args:
            *services (list): List of `porter.services.BaseService` instances
                to add to the model.

        Returns:
            None
        """
        for service in services:
            self.add_service(service)

    def add_service(self, service):
        """Add a service to the app from `service`.

        Args:
            service (object): Instance of `porter.services.BaseService`.

        Returns:
            None

        Raises:
            porter.exceptions.PorterError: If the type of
                `service` is not recognized.
        """
        if service.id in self._service_ids:
            raise exc.PorterError(
                f'a service has already been added using id={service.id}')
        self._services.append(service)
        self._service_ids.add(service.id)
        self.app.route(service.endpoint, **service.route_kwargs)(service)

    def run(self, *args, **kwargs):
        """
        Run the app.

        Args:
            *args: Positional arguments passed on to the wrapped `flask` app.
            **kwargs: Keyword arguments passed on to the wrapped `flask` app.
        """
        self.app.run(*args, **kwargs)

    def check_meta(self, meta):
        """Raise `ValueError` if `meta` contains invalid values, e.g. `meta`
        cannot be converted to JSON properly.

        Subclasses overriding this method should always use super() to call
        this method on the superclass unless they have a good reason not to.
        """
        try:
            # sort_keys=True tests the flask.jsonify implementation
            _ = json.dumps(meta, cls=cf.json_encoder, sort_keys=True)
        except TypeError:
            raise exc.PorterError(
                'Could not jsonify meta data. Make sure that meta data is '
                'valid JSON and that all keys are of the same type.')

    def _build_app(self):
        """Build and return the app.

        Any global properties of the app, such as error handling and response
        formatting, are added here.

        Returns:
            An instance of `api.App`.
        """
        app = api.App(__name__, static_folder=None)
        # register a custom JSON encoder
        app.json_encoder = cf.json_encoder
        # register error handler for all werkzeug default exceptions
        for error in werkzeug.exceptions.default_exceptions:
            app.register_error_handler(error, serve_error_message)
        app.register_error_handler(exc.PredictionError, serve_error_message)
        # This route that can be used to check if the app is running.
        # Useful for kubernetes/helm integration
        app.route('/', methods=['GET'])(serve_root)
        app.route(cn.LIVENESS_ENDPOINT, methods=['GET'])(ServeAlive(self))
        app.route(cn.READINESS_ENDPOINT, methods=['GET'])(ServeReady(self))
        return app
