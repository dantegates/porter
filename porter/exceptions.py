class PorterError(Exception):
    """Base Exception class for error's raised by `porter`."""


class PorterBadRequest(PorterError):
    """Exception class to raise when the POST JSON is not valid for
    predicting.
    """
    code = 400


class PredictionError(PorterError):
    """Exception raised when an error occurs during prediction."""

    def __init__(self, *args, model_name, model_version, model_meta, **kwargs):
        self.model_name = model_name
        self.model_version = model_version
        self.model_meta = model_meta
        super().__init__(*args, **kwargs)


class RequestMissingFields(PorterBadRequest):
    """Exception raised when POST request is missing required fields."""
    def __init__(self, fields):
        super().__init__(
            'request payload is missing the following field(s): {}'
            .format(fields))


class RequestContainsNulls(PorterBadRequest):
    """Exception raised when POST request contains null values."""
    def __init__(self, fields):
        super().__init__(
            'request payload had null values in the following field(s): {}'
            .format(fields))
