"""Tools for building RESTful services that exposes machine learning models.

Building and running an app with the tools in this module is as simple as

1. Instantiating :class:`ModelApp`.
2. Instantiating a "service". E.g. instantiate :class:`PredictionService` for each
   model you wish to add to the service.
3. Use the service(s) created in 2. to add models to the app with either
   :meth:`ModelApp.add_service()` or :meth:`ModelApp.add_services()`.

For example,

    >>> model_app = ModelApp()
    >>> prediction_service1 = PredictionService(...)
    >>> prediction_service2 = PredictionService(...)
    >>> model_app.add_services(prediction_servie1, prediction_service2)

Now the model app can be run with ``model_app.run()`` for development, or as an
example of running the app in production ``$ gunicorn my_module:model_app``.
"""

import abc
import collections
import json
import logging
import os
import warnings

import flask
import pandas as pd
import werkzeug.exceptions

from . import api
from . import config as cf
from . import constants as cn
from . import exceptions as exc
from . import responses as porter_responses
from . import schemas


# alias for convenience
_ID = cn.PREDICTION_PREDICTIONS_KEYS.ID

_logger = logging.getLogger(__name__)



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


class ServeRoot(StatefulRoute):
    _message = 'Send POST requests to /&lt model-name &gt/prediction/'

    def __init__(self, app):
        self.app = app

    def __call__(self):
        if self.app.expose_docs:
            return flask.redirect(self.app.docs_url)
        return self._message


class ServeAlive(StatefulRoute):
    """Class for building stateful liveness routes.

    Args:
        app (object): A :class:`ModelApp` instance. Instances of this class inspect
            ``app`` when called to determine if the app is alive.
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
        app (object): A :class:`ModelApp` instance. Instances of this class inspect
            ``app`` when called to determine if the app is alive.
    """

    logger = logging.getLogger(__name__)

    def __init__(self, app):
        self.app = app

    def __call__(self):
        """Serve readiness response."""
        response = porter_responses.make_ready_response(self.app)
        self.logger.info(response.data)
        return response.jsonify()


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
        validate_request_data (bool): Whether to validate the request data or
            not. Applies to all HTTP methods and does nothing if
            :meth:`add_request_schema()` is never called.
        validate_response_data (bool): Whether to validate the response data
            or not.  Applies to all HTTP methods and does nothing if
            :meth:`add_response_schema()` is never called.

    Attributes:
        id (str): A unique ID for the service.
        name (str): The model name. The final routed endpoint is generally
            derived from this attribute.
        api_version (str): The service version.
        meta (dict): Additional meta data added to the response body. Optional.
        log_api_calls (bool): Log request and response and response data.
            Default is False.
        namespace (str): A namespace that the service belongs to.
        validate_request_data (bool): Whether to validate the request data or
            not. Applies to all HTTP methods and does nothing if
            :meth:`add_request_schema()` is never called.
        validate_response_data (bool): Whether to validate the response data
            or not.  Applies to all HTTP methods and does nothing if
            :meth:`add_response_schema()` is never called.
        action (str): ``str`` describing the action of the service, e.g.
            "prediction". Used to determine the final routed endpoint.
        endpoint (str): The endpoint where the service is exposed.
        request_schemas (dict): Dictionary mapping HTTP methods to instances
            of :class:`porter.schemas.RequestSchema`. Each ``RequestSchema``
            object is added from calls to :meth:`add_request_schema` and
            instantiated from the corresponding arguments.
        response_schemas(dict): Dictionary mapping HTTP methods to a list of
            :class:`porter.schemas.ResponseSchema`. Each ``ResponseSchema`` object
            is added from calls to :meth:`add_request_schema` and
            instantiated from the corresponding arguments.
    """
    _ids = set()
    _logger = logging.getLogger(__name__)
    _default_response_schemas = [
        ('POST', 422, schemas.model_context_error, None),
        ('POST', 500, schemas.model_context_error, None),
    ]
    # subclasses can override this to add additional defaults
    _service_default_schemas = []

    # TODO: do we really need to validate responses? I could be useful for testing
    # but we could also manually call .validate()
    def __init__(self, *, name, api_version, meta=None, log_api_calls=False,
                 namespace='', validate_request_data=False,
                 validate_response_data=False):
        self.name = name
        self.api_version = api_version
        self.meta = {} if meta is None else meta
        self.check_meta(self.meta)
        self.namespace = namespace
        self.validate_request_data = validate_request_data
        self.validate_response_data = validate_response_data
        # Assign endpoint and ID last so they can be determined from other
        # instance attributes. If the order of assignment changes here these
        # methods may attempt to access attributes that have not been set yet
        # and fail.
        self.endpoint = self.define_endpoint()
        self.id = self.define_id()
        self.meta = self.update_meta(self.meta)
        self.log_api_calls = log_api_calls

        # these are a public interface exposing user registered schemas
        self.request_schemas = {}
        self.response_schemas = {}
        # these are used internally for lookups at runtime
        self._request_schemas = {}
        self._response_schemas = {}
        # add response schemas explicitly returned (i.e. raised) by porter.        
        for schema in self._default_response_schemas:
            self.add_response_schema(*schema)
        for schema in self._service_default_schemas:
            self.add_response_schema(*schema)

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
            # TODO: re-raise as model context error
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

            if self.validate_response_data:
                schema = self._response_schemas.get((api.request_method(), response.status_code))
                if schema is not None:
                    # TODO: use json.loads?
                    schema.validate(response.raw_data)
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
        value of one of the functions in :mod:`porter.responses` or an instance of
        :class:`porter.responses.Response`).

        Custom subclasses may find it easier to return a native Python object
        such as a ``str`` or ``dict``, in such cases the object must be
        "jsonify-able".
        """

    @abc.abstractproperty
    def status(self):
        """Return ``str`` representing the status of the service."""

    @property
    def route_kwargs(self):
        """Keyword arguments to use when routing ``self.serve()``."""
        return {}

    @property
    @abc.abstractproperty
    def action(self):
        """``str`` describing the action of the service, e.g. "prediction".
        Used to determine the final routed endpoint.
        """

    def define_id(self):
        """Return a unique ID for the service. This is used to set the ``id``
        attribute.
        """
        return self.endpoint

    # TODO: use schemas to validate
    def check_meta(self, meta):
        """Raise ``ValueError`` if ``meta`` contains invalid values, e.g. ``meta``
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
        if value and value.endswith('/'):
            value = value[:-1]
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
        """Return POST data.

        The data will be the return value of ``porter.config.json_encoder``.

        If ``self.validate_request_data is True`` and a request schema has
        been defined the data will be validated against the schema.
        """
        data = api.request_json(force=True)
        if self.validate_request_data:
            schema = self._request_schemas.get('POST')
            if schema is not None:
                try:
                    schema.validate(data)
                except ValueError as err:
                    if err.args[0].startswith('Schema validation failed'):
                        raise exc.InvalidModelInput(*err.args)
                    else:
                        raise err
        return data

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

    def add_request_schema(self, method, api_obj, description=None):
        method = method.upper()
        self.request_schemas[method] = schemas.RequestSchema(api_obj, description)
        self._request_schemas[method] = api_obj

    def add_response_schema(self, method, status_code, api_obj, description=None):
        method = method.upper()
        self._response_schemas[(method, status_code)] = api_obj
        if not method in self.response_schemas:
            self.response_schemas[method] = []
        self.response_schemas[method].append(schemas.ResponseSchema(api_obj, status_code, description))


class PredictionService(BaseService):
    """
    A prediction service. Instances can be added to instances of :class:`ModelApp`
    to serve predictions.

    Args:
        name (str): The model name. The final routed endpoint will become
            "/<namespace>/<name>/<api version>/<action>/".
        api_version (str): The model API version. The final routed endpoint
            will become "/<namespace>/<name>/<api version>/<action>/".
        meta (dict): Additional meta data added to the response body. Optional.
        log_api_calls (bool): Log request and response and response data.
            Default is False.
        namespace (str): String identifying a namespace that the service belongs
            to. The final routed endpoint will become
            "/<namespace>/<name>/<api version>/<action>/". Default is "".
        action (str): ``str`` describing the action of the service. Used to
            determine the final routed endpoint. Defaults to "prediction". The
            final routed endpoint will become
            "/<namespace>/<name>/<api version>/<action>/".
        model (object): An object implementing the interface defined by
            :class:`porter.datascience.BaseModel`.
        preprocessor (object or None): An object implementing the interface
            defined by :class:`porter.datascience.BaseProcessor`. If not ``None``, the
            `.process()` method of this object will be called on the POST
            request data and its output will be passed to ``model.predict()``.
            Optional.
        postprocessor (object or None): An object implementing the interface
            defined by :class:`porter.datascience.BaseProcessor`. If not ``None``, the
            `.process()` method of this object will be called on the output of
            ``model.predict()`` and its return value will be used to populate
            the predictions returned to the user. Optional.
        allow_nulls (bool): Are nulls allowed in the POST request data? If
            ``False`` an error is raised when nulls are found. Optional.
        batch_prediction (bool): Whether or not batch predictions are
            supported or not. If ``True`` the API will accept an array of objects
            to predict on. If ``False`` the API will only accept a single object
            per request. Optional.
        additional_checks (callable): Raises
            :class:`porter.exceptions.InvalidModelInput` or subclass thereof
            if POST request is invalid. The signature should accept a single
            positional argument for the validated POST input parsed to a
            ``pandas.DataFrame``.
        feature_schema (:class:`porter.schemas.Object` or `None`): Description
            of an a single feature set. Can be used to validate inputs if
            ``validate_request_data=True`` and document the API if added to an
            instance of :class:`ModelApp` where ``expose_docs=True``.
        prediction_schema (:class:`porter.schemas.Object` or `None`):
            Description of a single model prediction. Can be used to validate
            outputs if ``validate_request_data=True`` and document the API if
            added to an instance of :class:`ModelApp` where
            ``expose_docs=True``.
        **kwargs: Keyword arguments passed on to :class:`BaseService`.

    Attributes:
        id (str): A unique ID for the model. Composed of ``name`` and ``api_version``.
        name (str): The model's name.
        meta (dict): Additional meta data added to the response body. Optional.
        log_api_calls (bool): Log request and response and response data.
            Default is False.
        namespace (str): String identifying a namespace that the service belongs
            to. The final routed endpoint will become
            "/<namespace>/<name>/<api version>/prediction/". Default is "".
        api_version (str): The model API version.
        action (str): ``str`` describing the action of the service. Used to
            determine the final routed endpoint. The final routed endpoint
            will become "/<namespace>/<name>/<api version>/<action>/".
        endpoint (str): The endpoint where the model predictions are exposed.
            This is computed as "/<name>/<api version>/prediction/".
        model (object): An object implementing the interface defined by
            :class:`porter.datascience.BaseModel`.
        preprocessor (object or None): An object implementing the interface
            defined by :class:`porter.datascience.BaseProcessor`. If not `None`, the
            ``.process()`` method of this object will be called on the POST
            request data and its output will be passed to ``model.predict()``.
            Optional.
        postprocessor (object or None): An object implementing the interface
            defined by :class:`porter.datascience.BaseProcessor`. If not `None`, the
            `.process()` method of this object will be called on the output of
            ``model.predict()`` and its return value will be used to populate
            the predictions returned to the user. Optional.
        allow_nulls (bool): Are nulls allowed in the POST request data? If
            ``False`` an error is raised when nulls are found. Optional.
        batch_prediction (bool): Whether or not the endpoint supports batch
            predictions or not. If ``True`` the API will accept an array of
            objects to predict on. If ``False`` the API will only accept a
            single object per request. Optional.
        additional_checks (callable): Raises :class:`porter.exceptions.InvalidModelInput`
            or subclass thereof if POST request is invalid.
        feature_schema (`porter.schemas.Object` or None): Description of an
            individual instance to be predicted on. Can be used to validate
            inputs if `validate_request_data=True` and document the API if
            added to an instance of `ModelApp` where `expose_docs=True`.
        prediction_schema (`porter.schemas.Object` or None): Description of an
            individual prediction returned to the user. Can be used to
            validate outputs if `validate_request_data=True` and document the
            API if added to an instance of `ModelApp` where
            `expose_docs=True`.

    """

    route_kwargs = {'methods': ['GET', 'POST'], 'strict_slashes': False}
    action = 'prediction'
    _service_default_schemas = [
        ('GET', 200, schemas.String(), None)
    ]

    def __init__(self, *, model, preprocessor=None, postprocessor=None,
                 allow_nulls=False, action=None, batch_prediction=False,
                 additional_checks=None, feature_schema=None,
                 prediction_schema=None, **kwargs):
        self.model = model
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.allow_nulls = allow_nulls
        self.batch_prediction = batch_prediction
        if additional_checks is not None and not callable(additional_checks):
            raise exc.PorterError('`additional_checks` must be callable')
        self.action = action or self.action
        self.additional_checks = additional_checks

        self._preprocess_model_input = self.preprocessor is not None
        self._postprocess_model_output = self.postprocessor is not None

        # need to do this before handling schemas
        super().__init__(**kwargs)

        self.feature_schema = feature_schema
        self.prediction_schema = prediction_schema
        if self.feature_schema is not None:
            self._add_feature_schema(self.feature_schema)
            self.feature_columns = list(self.feature_schema.properties.keys())
        else:
            self.feature_columns = None
        # if None, we'll add the default schema anyway
        self._add_prediction_schema(self.prediction_schema)

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
            :class:`porter.exceptions.ModelContextError`: Raised whenever an error
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
        # retrieve the data and validate the inputs. If
        # self.validate_request_data is True and a feature schema was
        # provided, the schema is vetted in get_post_data()
        X_input = self.get_post_data()
        if not self.allow_nulls or self.additional_checks is not None:
            self.check_request(X_input, self.allow_nulls, self.additional_checks)

        # Once the input data has been fully validated, extract the feature
        # columns (all features provided in ``feature_schema``) if provided.
        # This allows the user to fully anticipate what features are passed
        # to the preprocessor.
        if self.feature_columns:
            X_preprocessed = X_input[self.feature_columns]
        else:
            X_preprocessed = X_input

        # preprocess if user specified a preprocessor
        if self._preprocess_model_input:
            X_preprocessed = self.preprocessor.process(X_preprocessed)

        # get the predictions
        preds = self.model.predict(X_preprocessed)

        # postprocess
        if self._postprocess_model_output:
            preds = self.postprocessor.process(X_input, X_preprocessed, preds)

        # finally format the predictions and return
        if self.batch_prediction:
            response = porter_responses.make_batch_prediction_response(
                self, X_input[_ID], preds)
        else:
            response = porter_responses.make_prediction_response(
                self, X_input[_ID].iloc[0], preds[0])

        return response

    @classmethod
    def check_request(cls, X_input, allow_nulls=False, additional_checks=None):
        """Check the POST request data raising an error if a check fails.

        Checks include

        1. ``X`` does not contain nulls (only if allow_nulls == True).
        2. Any additional checks in the user defined ``additional_checks``.

        Args:
            X (``pandas.DataFrame``): A ``pandas.DataFrame`` created from the POST
                request.
            allow_nulls (bool): Whether nulls are allowed in ``X``. False by
                default.

        Returns:
            None

        Raises:
            :class:`porter.exceptions.RequestContainsNulls`: If the input contains nulls
                and ``allow_nulls`` is False.
            :class:`porter.exceptions.RequestMissingFields`: If the input is missing
                required fields.
            :class:`porter.exceptions.InvalidModelInput`: If user defined ``additional_checks``
                fails.
        """
        cls._default_checks(X_input, allow_nulls)
        # Only perform user checks after the standard checks have been passed.
        # This allows the user to assume that all columns are present and there
        # are no nulls present (if allow_nulls is False).
        if additional_checks is not None:
            additional_checks(X_input)

    @staticmethod
    def _default_checks(X, allow_nulls):
        # TODO: deprecate this as well?
        if not allow_nulls and X.isnull().any().any():
            null_counts = X.isnull().sum()
            null_columns = null_counts[null_counts > 0].index.tolist()
            raise exc.RequestContainsNulls(null_columns)

    def get_post_data(self):
        """Return data from the most recent POST request as a ``pandas.DataFrame``.

        Returns:
            ``pandas.DataFrame``. Each ``row`` represents a single instance to
            predict on. If ``self.batch_prediction`` is ``False`` the ``DataFrame``
            will only contain one ``row``.
        """
        data = super().get_post_data()
        if not self.batch_prediction:
            data = [data]
        return pd.DataFrame(data)

    def _add_feature_schema(self, user_schema):
        assert isinstance(user_schema, schemas.Object), '``feature_schema`` must be an Object'
        # add ID to schema
        request_schema = schemas.Object(
            properties={
                'id': schemas.Integer('An ID uniquely identifying each instance in the POST body.'),
                **user_schema.properties},
            reference_name=user_schema.reference_name)
        if self.batch_prediction:
            request_schema = schemas.Array(item_type=request_schema)
        self.add_request_schema('POST', request_schema)

    def _add_prediction_schema(self, user_schema):
        prediction_schema = schemas.Object(
            'Model output',
            properties={
                'id': schemas.Integer('An ID uniquely identifying each instance in the POST body'),
                'prediction': user_schema or schemas.Number('Model Prediction')
            },
            reference_name=getattr(user_schema, 'reference_name', None)
        )

        if self.batch_prediction:
            prediction_schema = schemas.Array(item_type=prediction_schema)

        response_schema = schemas.Object(
            properties={
                'request_id': schemas.request_id,
                'schemas.model_context': schemas.model_context,
                'predictions': prediction_schema
            }
        )

        self.add_response_schema('POST', 200, response_schema)


# TODO: deprecate
class PredictionServiceConfig(PredictionService):
    def __init__(self, *args, **kwargs):
        warnings.warn('PredictionServiceConfig is deprecated. Use PredictionService.')
        super().__init__(*args, **kwargs)


class ModelApp:
    """
    Abstraction used to simplify building REST APIs that expose predictive
    models.

    Essentially this class is a wrapper around an instance of ``flask.Flask``.

    Args:
        name (str): Name for the application. This will appear in the documentation
            if ``expose_docs=True``. Optional.
        description (str): Description of the application. This will appear in
            the documentation if ``expose_docs=True``. Optional.
        version (str): Version of the application. This will appear in the
            documentation if ``expose_docs=True``. Optional.
        meta (dict): Additional meta data added to the response body in health
            checks. Optional.
        expose_docs (bool): If ``True`` API documentation will be served at
            ``docs_url``. The documentation is built from the
            ``request_schemas`` and ``response_schemas`` attributes of
            services added to the instance. Default is ``False``.
        docs_url (str): Endpoint for the API documentation. Ignored if
            ``expose_docs=False``. Defaults to '/docs/'

    Attributes:
        name (str): Name for the application.
        description (str): Description of the application.
        version (str): Version of the application.
        meta (dict): Additional meta data added to the response body in health
            checks.
        expose_docs (bool): Whether the instance is configured to expose API
            documentation.
        docs_url (str): Endpoint the API documentation is exposed at.
    """

    def __init__(self, *, name=__name__, description=None, version=None, meta=None,
                 expose_docs=False, docs_url='/docs/', docs_json_url='/_docs.json'):
        self.name = name
        self.meta = {} if meta is None else meta
        self.description = description
        self.version = version
        self.check_meta(self.meta)
        self.expose_docs = expose_docs
        self.docs_url = docs_url
        self.docs_json_url = docs_json_url

        self._services = []
        self._request_schemas = {}
        self._response_schemas = {}
        self._additional_params = {}
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
            *services (list): List of :class:`porter.services.BaseService` instances
                to add to the model.

        Returns:
            None
        """
        for service in services:
            self.add_service(service)

    def add_service(self, service):
        """Add a service to the app from ``service``.

        Args:
            service (object): Instance of :class:`porter.services.BaseService`.

        Returns:
            None

        Raises:
            :class:`porter.exceptions.PorterError`: If the type of
                ``service`` is not recognized.
        """
        # register the service with the add
        if service.id in self._service_ids:
            raise exc.PorterError(
                f'a service has already been added using id={service.id}')
        self._services.append(service)
        self._service_ids.add(service.id)
        # register the service schemas
        self._request_schemas[service.endpoint] = service.request_schemas
        self._response_schemas[service.endpoint] = service.response_schemas
        # tag the services
        self._additional_params[service.endpoint] = {'GET': {'tags': [service.name]}}
        self._additional_params[service.endpoint]['POST'] = {'tags': [service.name]}
        # finally route
        self.app.route(service.endpoint, **service.route_kwargs)(service)

    def run(self, *args, **kwargs):
        """
        Run the app.

        Args:
            *args: Positional arguments passed on to the wrapped ``flask`` app.
            **kwargs: Keyword arguments passed on to the wrapped ``flask`` app.
        """
        # Because the ``ModelApp`` API allows services to be added to the app
        # after it's been instantiated we build the docs immediately before
        # starting the app to be sure that all services are included in the docs.
        if self.expose_docs:
            self._route_docs()
        self.app.run(*args, **kwargs)

    # TODO: use schemas to validate
    def check_meta(self, meta):
        """Raise ``ValueError`` if ``meta`` contains invalid values, e.g. ``meta``
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

    # TODO: we need tests for this
    def _build_app(self):
        """Build and return the app.

        Any global properties of the app, such as error handling and response
        formatting, are added here.

        Returns:
            An instance of :class:`porter.api.App`.
        """
        app = api.App(self.name, static_folder='porter/assets')

        # register a custom JSON encoder
        app.json_encoder = cf.json_encoder

        # register error handler for all werkzeug default exceptions
        for error in werkzeug.exceptions.default_exceptions:
            app.register_error_handler(error, serve_error_message)
        app.register_error_handler(exc.PredictionError, serve_error_message)

        # route the health checks
        serve_alive = ServeAlive(self)
        serve_ready = ServeReady(self)
        app.route(cn.LIVENESS_ENDPOINT, methods=['GET'])(serve_alive)
        app.route(cn.READINESS_ENDPOINT, methods=['GET'])(serve_ready)

        # register the schemas for the health checks
        if self.expose_docs:
            health_check_response = {'GET': [schemas.ResponseSchema(schemas.health_check, 200)]}
            self._response_schemas[cn.LIVENESS_ENDPOINT] = health_check_response
            self._response_schemas[cn.READINESS_ENDPOINT] = health_check_response
            self._additional_params[cn.LIVENESS_ENDPOINT] = {'GET': {'tags': ['Health Check']}}
            self._additional_params[cn.READINESS_ENDPOINT] = {'GET': {'tags': ['Health Check']}}

        # route root
        serve_root = ServeRoot(self)
        app.route('/', methods=['GET'])(serve_root)

        return app

    def _route_docs(self):
        openapi_json =  schemas.make_openapi_spec(self.name, self.description, self.version,
                                          self._request_schemas, self._response_schemas,
                                          self._additional_params)

        @self.app.route('/assets/swagger-ui/{filename}')
        def swagger_ui(filename):
            return self.app.send_static_file(filename)

        @self.app.route(self.docs_json_url)
        def docs_json():
            return json.dumps(openapi_json)

        @self.app.route(self.docs_url)
        def docs():
            html = schemas.make_docs_html(self.docs_json_url)
            return html
