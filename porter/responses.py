import traceback

import flask

from . import constants as cn


# alias for convenience
_IS_READY = cn.HEALTH_CHECK.VALUES.STATUS_IS_READY


# NOTE: private functions make testing easier as they bypass `flask.jsonify`


def make_prediction_response(model_name, model_version, model_meta, id_keys, predictions):
    payload = _make_prediction_payload(model_name, model_version, model_meta,
                                       id_keys, predictions)
    return flask.jsonify(payload)


def _make_prediction_payload(model_name, model_version, model_meta, id_keys, predictions):
    payload = {
        cn.PREDICTION.KEYS.MODEL_NAME: model_name,
        cn.PREDICTION.KEYS.MODEL_VERSION: model_version,
        cn.PREDICTION.KEYS.PREDICTIONS: [
            {
                cn.PREDICTION.KEYS.ID: id,
                cn.PREDICTION.KEYS.PREDICTION: p
            }
            for id, p in zip(id_keys, predictions)]
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
    return {
        cn.ERROR_KEYS.ERROR: type(error).__name__,
        # getattr() is used to work around werkzeug's bad implementation
        # of HTTPException (i.e. HTTPException inherits from Exception but
        # exposes a different API, namely
        # Exception.message -> HTTPException.description).
        cn.ERROR_KEYS.MESSAGE: getattr(error, 'description', error.args),
        cn.ERROR_KEYS.TRACEBACK: traceback.format_exc(),
        cn.ERROR_KEYS.USER_DATA: user_data}


def make_alive_response(app_state):
    return flask.jsonify(app_state)


def make_ready_response(app_state):
    ready = _is_ready(app_state)
    response = flask.jsonify(app_state)
    response.status_code = 200 if ready else 503  # service unavailable
    return response


def _is_ready(app_state):
    services = app_state[cn.HEALTH_CHECK.KEYS.SERVICES]
    # app must define services and all services must be ready
    return services and all(
        svc[cn.HEALTH_CHECK.KEYS.STATUS] is _IS_READY for svc in services.values())
