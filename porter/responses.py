import traceback

import flask

from .constants import KEYS, APP


def make_prediction_response(model_id, id_keys, predictions):
    payload = _make_prediction_payload(model_id, id_keys, predictions)
    return flask.jsonify(payload)


def _make_prediction_payload(model_id, id_keys, predictions):
    return {
        KEYS.PREDICTION.MODEL_ID: model_id,
        KEYS.PREDICTION.PREDICTIONS: [
            {
                KEYS.PREDICTION.ID: id,
                KEYS.PREDICTION.PREDICTION: p
            }
            for id, p in zip(id_keys, predictions)]
    }


def make_error_response(error):
    payload = _make_error_payload(error)
    response = flask.jsonify(payload)
    response.status_code = getattr(error, 'code', 500)
    return response


def _make_error_payload(error):
    return {
        KEYS.ERROR.ERROR: type(error).__name__,
        # getattr() is used to work around werkzeug's bad implementation
        # of HTTPException (i.e. HTTPException inherits from Exception but
        # exposes a different API, namely
        # Exception.message -> HTTPException.description).
        KEYS.ERROR.MESSAGE: getattr(error, 'description', error.args),
        KEYS.ERROR.TRACEBACK: traceback.format_exc()}


def make_alive_response(app_state):
    return flask.jsonify(app_state)


def make_ready_response(app_state):
    ready = _is_ready(app_state)
    response = flask.jsonify(app_state)
    response.status_code = 200 if ready else 503  # service unavailable
    return response


def _is_ready(app_state):
    services = app_state[APP.STATE.SERVICES]
    # app must define services and all services must be ready
    return services and all(
        svc[APP.STATE.STATUS] == APP.STATE.READY for svc in services.values())
