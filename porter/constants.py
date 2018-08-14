from .utils import Endpoint


class ERROR_KEYS:
    """Keys for for error responses.

    Attributes:
        NAME: Name of error key.
        MESSAGES: Error messages key.
        TRACEBACK: Error traceback key.
        USER_DATA: Key for user data.
    """
    NAME = 'name'
    MESSAGES = 'messages'
    TRACEBACK = 'traceback'
    USER_DATA = 'user_data'


class PREDICTION(Endpoint):
    """Container prediction endpoint constants."""

    ENDPOINT_TEMPLATE = '/{model_name}/prediction'

    class KEYS:
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


class HEALTH_CHECK(Endpoint):
    class KEYS:
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
        MODEL_VERSION = 'version'
        META = 'meta'
        PORTER_VERSION = 'porter_version'

    class VALUES:
        STATUS_IS_READY = 'READY'


class LIVENESS(HEALTH_CHECK):
    ENDPOINT = '/-/alive'


class READINESS(HEALTH_CHECK):
    ENDPOINT = '/-/ready'
