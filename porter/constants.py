import datetime


_REQUEST_ID = 'request_id'


class ERRORS:
    """Container for error response constants."""
    class RESPONSE:
        class KEYS:
            """Keys for for error responses.

            Attributes:
                NAME: Name of error key.
                MESSAGES: Error messages key.
                TRACEBACK: Error traceback key.
                USER_DATA: Key for user data.
            """
            ERROR = 'error'
            REQUEST_ID = _REQUEST_ID
            NAME = 'name'
            MESSAGES = 'messages'
            TRACEBACK = 'traceback'
            USER_DATA = 'user_data'


class PREDICTION:
    """Container prediction endpoint constants."""

    ENDPOINT_TEMPLATE = '/{model_name}/{api_version}/prediction'

    class RESPONSE:
        class KEYS:
            """Keys for requests received/sent from /<model>/prediction.

            Attributes:
                MODEL_NAME: Model name.
                API_VERSION: API version.
                PREDICTIONS: Array of predictions key.
                ID: Unique record ID key.
                PREDICTION: Prediction for a given ID key.
            """
            MODEL_NAME = 'model_name'
            API_VERSION = 'api_version'
            ID = 'id'
            PREDICTIONS = 'predictions'
            PREDICTION = 'prediction'
            ERROR = ERRORS.RESPONSE.KEYS.ERROR
            REQUEST_ID = _REQUEST_ID


class BATCH_PREDICTION:
    """Container prediction endpoint constants."""

    ENDPOINT_TEMPLATE = '/{model_name}/{api_version}/batchPrediction'
    class RESPONSE:
        class KEYS:
            # meta key
            MODEL_ENDPOINT = 'model_endpoint'
            MAX_WORKERS = 'max_workers'

class HEALTH_CHECK:
    """Base class for health check endpoint constants."""

    class RESPONSE:
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
            API_VERSION = 'api_version'
            META = 'meta'
            PORTER_VERSION = 'porter_version'
            DEPLOYED_ON = 'deployed_on'
            APP_META = 'app_meta'

        class VALUES:
            STATUS_IS_READY = 'READY'
            DEPLOYED_ON = datetime.datetime.now().isoformat()


class LIVENESS(HEALTH_CHECK):
    """Container for liveness endpoint constants."""
    ENDPOINT = '/-/alive'


class READINESS(HEALTH_CHECK):
    """Container for readiness endpoint constants."""
    ENDPOINT = '/-/ready'
