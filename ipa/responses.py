import traceback


class BaseResponse(dict):
    def __init__(self, *args, **kwargs):
        response = self.make_response(*args, **kwargs)
        super(BaseResponse, self).__init__(response)


class PredictionResponse(BaseResponse):
    @staticmethod
    def make_response(model_id, id_keys, predictions):
        return {
            'model_id': model_id,
            'predictions': [{id: p} for id, p in zip(id_keys, predictions)]
        }


class ErrorResponse(BaseResponse):
    @staticmethod
    def make_response(error):
        return {
            'error': type(error).__name__,
            # getattr() is used to work around werkzeug's bad implementation
            # of HTTPException (i.e. HTTPException inherits from Exception but
            # exposes a different API, namely
            # Exception.message -> HTTPException.description).
            'message': getattr(error, 'description', error.message),
            'traceback': traceback.format_exc()}
