class KEYS:
    """Container for key names of JSON objects for app input/output."""

    class PREDICTION:
        """Keys for requests received/sent from /<model>/prediction.

        Attributes:
            MODEL_NAME: Model name.
            MODEL_VERSION: Model version.
            PREDICTIONS: Array of predictions key.
            ID: Unique record ID key.
            PREDICTION: Prediction for a given ID key.
        """
        MODEL_NAME = 'model_name'
        MODEL_VERSION = 'model_version'
        ID = 'id'
        PREDICTIONS = 'predictions'
        PREDICTION = 'prediction'

    class ERROR:
        """Keys for for error responses.

        Attributes:
            ERROR: Name of error key.
            MESSAGE: Error message key.
            TRACEBACK: Error traceback key.
        """
        ERROR = 'error'
        MESSAGE = 'message'
        TRACEBACK = 'traceback'


class ENDPOINTS:
    """Container for endpoints."""
    PREDICTION_TEMPLATE = '/{model_name}/prediction'
    LIVENESS = '/-/alive'
    READINESS = '/-/ready'


class APP:
    """Container for app constants."""

    class STATE:
        """Container for keys and values of an applications state.

        Attributes:
            SERVICES: The key to access all services added to an instance of
                `ModelApp`.
            STATUS: The key to access a services stats.
            READY: The value indicating that a service is ready
        """
        SERVICES = 'services'
        STATUS = 'status'
        ENDPOINT = 'endpoint'
        NAME = 'name'
        VERSION = 'version'
        META = 'meta'
        READY = 'READY'
