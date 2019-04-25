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


class Response:
    def __init__(self, data, status_code=None):
        self.data = data
        self.status_code = status_code

    def jsonify(self):
        jsonified = api.jsonify(self.data)
        if self.status_code is not None:
            jsonified.status_code = self.status_code
        return jsonified


def make_prediction_response(model_name, api_version, model_meta, id_keys,
                             predictions, batch_prediction):
    if batch_prediction:
        payload = _make_batch_prediction_payload(model_name, api_version, model_meta,
                                                 id_keys, predictions)
    else:
        payload = _make_single_prediction_payload(model_name, api_version, model_meta,
                                                  id_keys, predictions)
    return Response(payload)


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
    return Response(objects)


def make_error_response(error, *, include_message, include_traceback, include_user_data):
    # silent=True -> flask.request.get_json(...) returns None if user did not
    # provide data
    user_data = api.request_json(silent=True, force=True) if include_user_data else None
    request_id = api.request_id()
    payload = _make_error_payload(
        error,
        request_id,
        user_data=user_data,
        include_message=include_message,
        include_traceback=include_traceback,
        include_user_data=include_user_data)
    response = Response(payload, getattr(error, 'code', 500))
    return response


def _make_error_payload(error, request_id, *, user_data, include_message,
                        include_traceback, include_user_data):
    payload = {}
    payload[_ERROR_KEYS.ERROR] = error_dict = {}
    # all errors should at least return the name and a request ID for debugging
    error_dict[_ERROR_KEYS.NAME] = type(error).__name__
    error_dict[_ERROR_KEYS.REQUEST_ID] = request_id
    # include optional attributes
    if include_message:
        # getattr() is used to work around werkzeug's bad implementation of
        # HTTPException (i.e. HTTPException inherits from Exception but exposes a
        # different API, namely Exception.message -> HTTPException.description).
        messages = [error.description] if hasattr(error, 'description') else error.args
        error_dict[_ERROR_KEYS.MESSAGES] = messages
    if include_traceback:
        error_dict[_ERROR_KEYS.TRACEBACK] = traceback.format_exc()
    if include_user_data:
        error_dict[_ERROR_KEYS.USER_DATA] = user_data

    # if the error was generated while predicting add model meta data to error
    # message - note that isinstance(obj, cls) is True if obj is an instance
    # of a subclass of cls
    if isinstance(error, exc.ModelContextError):
        payload[_PREDICTION_KEYS.MODEL_NAME] = error.model_name
        payload[_PREDICTION_KEYS.API_VERSION] = error.api_version
        payload.update(error.model_meta)

    return payload


def make_alive_response(app_state):
    return Response(app_state)


def make_ready_response(app_state):
    ready = _is_ready(app_state)
    response = Response(app_state, 200 if ready else 503)
    return response


def _is_ready(app_state):
    services = app_state[_HEALTH_CHECK_KEYS.SERVICES]
    # app must define services and all services must be ready
    return services and all(svc[_HEALTH_CHECK_KEYS.STATUS] is _IS_READY
                            for svc in services.values())
