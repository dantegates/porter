import traceback


class BasePayload(dict):
    def __init__(self, *args, **kwargs):
        payload = self._init_payload(*args, **kwargs)
        super(BasePayload, self).__init__(payload)


class PredictionPayload(BasePayload):
    @staticmethod
    def _init_payload(model_id, id_keys, predictions):
        return {
            'model_id': model_id,
            'predictions': [{id: p} for id, p in zip(id_keys, predictions)]
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


def make_prediction_response(model_id, id_keys, predictions, status_code):
    payload = PredictionPayload(model_id, id_keys, predictions)
    return flask.jsonify(payload)


def make_error_response(error):
    paylod = {
        'error': type(error).__name__,
        # getattr() is used to work around werkzeug's bad implementation
        # of HTTPException (i.e. HTTPException inherits from Exception but
        # exposes a different API, namely
        # Exception.message -> HTTPException.description).
        'message': getattr(error, 'description', error.message),
        'traceback': traceback.format_exc()}
    response = flask.jsonify(payload)
    response.status_code = getattr(error, 'code', 500)
    return response
