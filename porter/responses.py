import traceback

import flask

from . import constants as cn
from . import exceptions as exc

# alias for convenience
_IS_READY = cn.HEALTH_CHECK.RESPONSE.VALUES.STATUS_IS_READY


# NOTE: private functions make testing easier as they bypass `flask` methods
# that require a context, e.g. `flask.jsonify`


def make_prediction_response(model_name, model_version, model_meta, id_keys,
                             predictions, batch_prediction):
    if batch_prediction:
        payload = _make_batch_prediction_payload(model_name, model_version, model_meta,
                                                 id_keys, predictions)
    else:
        payload = _make_single_prediction_payload(model_name, model_version, model_meta,
                                                  id_keys, predictions)
    return flask.jsonify(payload)


def _make_batch_prediction_payload(model_name, model_version, model_meta, id_keys, predictions):
    payload = {
        cn.PREDICTION.RESPONSE.KEYS.MODEL_NAME: model_name,
        cn.PREDICTION.RESPONSE.KEYS.MODEL_VERSION: model_version,
        cn.PREDICTION.RESPONSE.KEYS.PREDICTIONS: [
            {
                cn.PREDICTION.RESPONSE.KEYS.ID: id,
                cn.PREDICTION.RESPONSE.KEYS.PREDICTION: p
            }
            for id, p in zip(id_keys, predictions)]
    }
    payload.update(model_meta)
    return payload


def _make_single_prediction_payload(model_name, model_version, model_meta, id_keys, predictions):
    payload = {
        cn.PREDICTION.RESPONSE.KEYS.MODEL_NAME: model_name,
        cn.PREDICTION.RESPONSE.KEYS.MODEL_VERSION: model_version,
        cn.PREDICTION.RESPONSE.KEYS.PREDICTIONS:
            {
                cn.PREDICTION.RESPONSE.KEYS.ID: id_keys[0],
                cn.PREDICTION.RESPONSE.KEYS.PREDICTION: predictions[0]
            }
    }
    payload.update(model_meta)
    return payload


def make_error_response(error):
    # silent=True -> flask.request.get_json(...) returns None if user did not
    # provide data
    user_data = flask.request.get_json(silent=True, force=True)
    payload = _make_error_payload(error, user_data)
    response = flask.jsonify(payload)
    response.status_code = getattr(error, 'code', 500)
    return response


def _make_error_payload(error, user_data):
    payload = {}
    # if the error was generated while predicting add model meta data to error
    # message
    if isinstance(error, exc.PorterPredictionError):
        payload[cn.PREDICTION.RESPONSE.KEYS.MODEL_NAME] = error.model_name
        payload[cn.PREDICTION.RESPONSE.KEYS.MODEL_VERSION] = error.model_version
        payload.update(error.model_meta)
    # getattr() is used to work around werkzeug's bad implementation of
    # HTTPException (i.e. HTTPException inherits from Exception but exposes a
    # different API, namely Exception.message -> HTTPException.description).
    messages = [error.description] if hasattr(error, 'description') else error.args
    payload[cn.ERRORS.RESPONSE.KEYS.ERROR] = {
        cn.ERRORS.RESPONSE.KEYS.NAME: type(error).__name__,
        cn.ERRORS.RESPONSE.KEYS.MESSAGES: messages,
        cn.ERRORS.RESPONSE.KEYS.TRACEBACK: traceback.format_exc(),
        cn.ERRORS.RESPONSE.KEYS.USER_DATA: user_data}
    return payload


def make_alive_response(app_state):
    return flask.jsonify(app_state)


def make_ready_response(app_state):
    ready = _is_ready(app_state)
    response = flask.jsonify(app_state)
    response.status_code = 200 if ready else 503  # service unavailable
    return response


def _is_ready(app_state):
    services = app_state[cn.HEALTH_CHECK.RESPONSE.KEYS.SERVICES]
    # app must define services and all services must be ready
    return services and all(svc[cn.HEALTH_CHECK.RESPONSE.KEYS.STATUS] is _IS_READY
                            for svc in services.values())
