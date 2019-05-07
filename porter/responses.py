import traceback

from . import __version__ as VERSION
from . import config as cf
from . import constants as cn
from . import exceptions as exc
from . import api

# aliases for convenience
_IS_READY = cn.HEALTH_CHECK.RESPONSE.VALUES.STATUS_IS_READY
_PREDICTION_KEYS = cn.PREDICTION.RESPONSE.KEYS
_ERROR_KEYS = cn.ERRORS.RESPONSE.KEYS
_HEALTH_CHECK_KEYS = cn.HEALTH_CHECK.RESPONSE.KEYS


# NOTE: private functions make testing easier as they bypass `flask` methods
# that require a context, e.g. `api.jsonify`


class Response:
    def __init__(self, data, status_code=None):
        self.data = data
        self.status_code = status_code

    def jsonify(self):
        jsonified = api.jsonify(self.data)
        if self.status_code is not None:
            jsonified.status_code = self.status_code
        return jsonified


def make_prediction_response(model_service, id_value, prediction):
    payload = {}
    payload[_PREDICTION_KEYS.MODEL_CONTEXT] = _init_model_context(model_service)
    payload[_PREDICTION_KEYS.PREDICTIONS] = {
        _PREDICTION_KEYS.ID: id_value,
        _PREDICTION_KEYS.PREDICTION: prediction

    }
    return Response(payload)


def make_batch_prediction_response(model_service, id_values, predictions):
    payload = {}
    payload[_PREDICTION_KEYS.MODEL_CONTEXT] = _init_model_context(model_service)
    payload[_PREDICTION_KEYS.PREDICTIONS] = [
        {
            _PREDICTION_KEYS.ID: id,
            _PREDICTION_KEYS.PREDICTION: p
        }
        for id, p in zip(id_values, predictions)
    ]
    return Response(payload)


def _init_model_context(model_service):
    payload = {
        _PREDICTION_KEYS.MODEL_NAME: model_service.name,
        _PREDICTION_KEYS.API_VERSION: model_service.api_version,
    }
    if cf.return_request_id_with_prediction:
        payload[_PREDICTION_KEYS.REQUEST_ID] = api.request_id()
    payload[_PREDICTION_KEYS.MODEL_META] = model_service.meta
    return payload


def make_middleware_response(objects):
    return Response(objects)


def make_error_response(error):
    payload = {}
    payload[_ERROR_KEYS.ERROR] = error_dict = {}

    # all errors should at least return the name
    error_dict[_ERROR_KEYS.NAME] = type(error).__name__

    # include optional attributes

    ## these are "top-level" attributes
    if cf.return_request_id_on_error:
        payload[_ERROR_KEYS.REQUEST_ID] = api.request_id()

    # if the error was generated while predicting add model meta data to error
    # message - note that isinstance(obj, cls) is True if obj is an instance
    # of a subclass of cls
    if isinstance(error, exc.ModelContextError):
        payload[_ERROR_KEYS.MODEL_CONTEXT] = _init_model_context(error.model_service)

    ## these are "error specific" attributes
    if cf.return_message_on_error:
        # getattr() is used to work around werkzeug's bad implementation of
        # HTTPException (i.e. HTTPException inherits from Exception but exposes a
        # different API, namely Exception.message -> HTTPException.description).
        messages = [error.description] if hasattr(error, 'description') else error.args
        error_dict[_ERROR_KEYS.MESSAGES] = messages

    if cf.return_traceback_on_error:
        error_dict[_ERROR_KEYS.TRACEBACK] = traceback.format_exc()

    if cf.return_user_data_on_error:
        # silent=True -> flask.request.get_json(...) returns None if user did not
        error_dict[_ERROR_KEYS.USER_DATA] = api.request_json(silent=True, force=True)

    return Response(payload, getattr(error, 'code', 500))


def make_alive_response(app):
    app_state = _build_app_state(app)
    return Response(app_state)


def make_ready_response(app):
    app_state = _build_app_state(app)
    ready = _is_ready(app_state)
    response = Response(app_state, 200 if ready else 503)
    return response


def _is_ready(app_state):
    services = app_state[_HEALTH_CHECK_KEYS.SERVICES]
    print(app_state)
    # app must define services and all services must be ready
    return services and all(svc[_HEALTH_CHECK_KEYS.STATUS] is _IS_READY
                            for svc in services.values())


def _build_app_state(app):
    """Return the app state as a "jsonify-able" object."""
    return {
        _HEALTH_CHECK_KEYS.PORTER_VERSION: VERSION,
        _HEALTH_CHECK_KEYS.DEPLOYED_ON: cn.HEALTH_CHECK.RESPONSE.VALUES.DEPLOYED_ON,
        _HEALTH_CHECK_KEYS.APP_META: app.meta,
        _HEALTH_CHECK_KEYS.SERVICES: {
            service.id: {
                _HEALTH_CHECK_KEYS.MODEL_CONTEXT: _init_model_context(service),
                _HEALTH_CHECK_KEYS.STATUS: service.status,
                _HEALTH_CHECK_KEYS.ENDPOINT: service.endpoint,
            }
            for service in app._services
        }
    }
 