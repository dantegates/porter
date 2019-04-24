import traceback

from . import api
from . import constants as cn
from . import exceptions as exc

# aliases for convenience
_IS_READY = cn.HEALTH_CHECK.RESPONSE.VALUES.STATUS_IS_READY
_PREDICTION_KEYS = cn.PREDICTION.RESPONSE.KEYS
_ERROR_KEYS = cn.ERRORS.RESPONSE.KEYS
_HEALTH_CHECK_KEYS = cn.HEALTH_CHECK.RESPONSE.KEYS


# NOTE: private functions make testing easier as they bypass `flask` methods
# that require a context, e.g. `api.jsonify`


def make_prediction_response(model_name, api_version, model_meta, id_keys,
                             predictions, batch_prediction):
    if batch_prediction:
        payload = _make_batch_prediction_payload(model_name, api_version, model_meta,
                                                 id_keys, predictions)
    else:
        payload = _make_single_prediction_payload(model_name, api_version, model_meta,
                                                  id_keys, predictions)
    return api.jsonify(payload)


def _make_batch_prediction_payload(model_name, api_version, model_meta, id_keys, predictions):
    payload = {
        _PREDICTION_KEYS.MODEL_NAME: model_name,
        _PREDICTION_KEYS.API_VERSION: api_version,
        _PREDICTION_KEYS.PREDICTIONS: [
            {
                _PREDICTION_KEYS.ID: id,
                _PREDICTION_KEYS.PREDICTION: p
            }
            for id, p in zip(id_keys, predictions)]
    }
    payload.update(model_meta)
    return payload


def _make_single_prediction_payload(model_name, api_version, model_meta, id_keys, predictions):
    payload = {
        _PREDICTION_KEYS.MODEL_NAME: model_name,
        _PREDICTION_KEYS.API_VERSION: api_version,
        _PREDICTION_KEYS.PREDICTIONS:
            {
                _PREDICTION_KEYS.ID: id_keys[0],
                _PREDICTION_KEYS.PREDICTION: predictions[0]
            }
    }
    payload.update(model_meta)
    return payload


def make_middleware_response(objects):
    return api.jsonify(objects)


def make_error_response(error):
    # silent=True -> flask.request.get_json(...) returns None if user did not
    # provide data
    user_data = api.request_json(silent=True, force=True)
    payload = _make_error_payload(error, user_data)
    response = api.jsonify(payload)
    response.status_code = getattr(error, 'code', 500)
    return response


def _make_error_payload(error, user_data):
    payload = {}
    # if the error was generated while predicting add model meta data to error
    # message - note that isinstance(obj, cls) is True if obj is an instance
    # of a subclass of cls
    if isinstance(error, exc.ModelContextError):
        payload[_PREDICTION_KEYS.MODEL_NAME] = error.model_name
        payload[_PREDICTION_KEYS.API_VERSION] = error.api_version
        payload.update(error.model_meta)
    # getattr() is used to work around werkzeug's bad implementation of
    # HTTPException (i.e. HTTPException inherits from Exception but exposes a
    # different API, namely Exception.message -> HTTPException.description).
    messages = [error.description] if hasattr(error, 'description') else error.args
    payload[_ERROR_KEYS.ERROR] = {
        _ERROR_KEYS.NAME: type(error).__name__,
        _ERROR_KEYS.MESSAGES: messages,
        _ERROR_KEYS.TRACEBACK: traceback.format_exc(),
        _ERROR_KEYS.USER_DATA: user_data}
    return payload


def make_alive_response(app_state):
    return api.jsonify(app_state)


def make_ready_response(app_state):
    ready = _is_ready(app_state)
    response = api.jsonify(app_state)
    response.status_code = 200 if ready else 503  # service unavailable
    return response


def _is_ready(app_state):
    services = app_state[_HEALTH_CHECK_KEYS.SERVICES]
    # app must define services and all services must be ready
    return services and all(svc[_HEALTH_CHECK_KEYS.STATUS] is _IS_READY
                            for svc in services.values())
