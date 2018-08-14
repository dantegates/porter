class PorterError(Exception):
    """Base Exception class for error's raised by `porter`."""


class PredictionError(PorterError):
    """Exception raised when an error occurs during prediction."""

    def __init__(self, *args, model_name, model_version, model_meta, **kwargs):
        self.model_name = model_name
        self.model_version = model_version
        self.model_meta = model_meta
        super().__init__(*args, **kwargs)
