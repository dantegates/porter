"""Global constants defining endpoint naming conventions, etc."""

import datetime
import os

# this must be an absolute path
ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'assets')


_MODEL_CONTEXT = 'model_context'
_PREDICTIONS = 'predictions'


LIVENESS_ENDPOINT = '/-/alive'
READINESS_ENDPOINT = '/-/ready'
ENDPOINT_TEMPLATE = '{namespace}/{service_name}/{api_version}/{action}'


class BASE_KEYS:
    REQUEST_ID = 'request_id'


class HEALTH_CHECK_KEYS(BASE_KEYS):
    PORTER_VERSION = 'porter_version'
    DEPLOYED_ON = 'deployed_on'
    APP_META = 'app_meta'
    SERVICES = 'services'


class HEALTH_CHECK_VALUES:
    IS_READY = 'READY'
    DEPLOYED_ON = datetime.datetime.now().isoformat()


class HEALTH_CHECK_SERVICES_KEYS:
    ENDPOINT = 'endpoint'
    STATUS = 'status'
    MODEL_CONTEXT = _MODEL_CONTEXT


class GENERIC_ERROR_KEYS(BASE_KEYS):
    ERROR = 'error'


class MODEL_CONTEXT_ERROR_KEYS:
    MODEL_CONTEXT = 'model_context'


class ERROR_BODY_KEYS:
    NAME = 'name'
    MESSAGES = 'messages'
    TRACEBACK = 'traceback'
    USER_DATA = 'user_data'


class MODEL_CONTEXT_KEYS:
    MODEL_NAME = 'model_name'
    API_VERSION = 'api_version'
    MODEL_META = 'model_meta'


class PREDICTION_KEYS(BASE_KEYS):
    MODEL_CONTEXT = _MODEL_CONTEXT
    PREDICTIONS = _PREDICTIONS 


class PREDICTION_PREDICTIONS_KEYS:
    ID = 'id'
    PREDICTION = 'prediction'
