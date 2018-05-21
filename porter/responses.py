import traceback

import flask


class BasePayload(dict):
    def __init__(self, *args, **kwargs):
        payload = self._init_payload(*args, **kwargs)
        super(BasePayload, self).__init__(payload)

    def _init_payload(self):
        raise NotImplementedError


class PredictionPayload(BasePayload):
    @staticmethod
    def _init_payload(model_id, id_keys, predictions):
        return {
            'model_id': model_id,
            'predictions': [{"id": id, "prediction": p}
                            for id, p in zip(id_keys, predictions)]
        }


class ErrorPayload(BasePayload):
    @staticmethod
    def _init_payload(error):
        return {
            'error': type(error).__name__,
            # getattr() is used to work around werkzeug's bad implementation
            # of HTTPException (i.e. HTTPException inherits from Exception but
            # exposes a different API, namely
            # Exception.message -> HTTPException.description).
            'message': getattr(error, 'description', error.message),
            'traceback': traceback.format_exc()}


def make_prediction_response(model_id, id_keys, predictions):
    payload = PredictionPayload(model_id, id_keys, predictions)
    return flask.jsonify(payload)


def make_error_response(error):
    payload = ErrorPayload(error)
    response = flask.jsonify(payload)
    response.status_code = getattr(error, 'code', 500)
    return response
