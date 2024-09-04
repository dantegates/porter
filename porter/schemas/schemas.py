from .. import config as cf
from .. import constants as cn
from . import openapi


# define base objects
#====================


request_id = openapi.String(
    'Hex value of UUID assigned to the request.',
    reference_name='RequestID')

app_meta = openapi.Object(properties={}, additional_params=dict(additionalProperties=True))

model_meta = openapi.Object(properties={}, additional_params=dict(additionalProperties=True))

model_context = openapi.Object(
    properties={
        cn.MODEL_CONTEXT_KEYS.MODEL_NAME: openapi.String('The name of the model.'),
        cn.MODEL_CONTEXT_KEYS.API_VERSION: openapi.String('The model API version.'),
        cn.MODEL_CONTEXT_KEYS.MODEL_META: model_meta
    },
    reference_name='ModelContext'
)


error_messages = openapi.Array('An array of messages describing the error.', item_type=openapi.String())
error_name = openapi.String('Name of the error')
error_traceback = openapi.String('The error traceback')


# define response objects determined by app configurations
#=========================================================

_base_response = {}
if cf.return_request_id:
    _base_response = {'request_id': request_id}


_error_body = {'name': error_name}
if cf.return_message_on_error:
    _error_body['messages'] = error_messages
if cf.return_traceback_on_error:
    _error_body['traceback'] = error_traceback
# user data could be anything? it's only recommended for development anyway
# if cf.return_user_data_on_error:
#     _error_body['user_data'] = ?


# define the final response objects
#==================================


health_check = openapi.Object(
    'Description of the applications status. Useful for load balancing and debugging',
    properties={
        **_base_response,
        cn.HEALTH_CHECK_KEYS.PORTER_VERSION: openapi.String('The version of the porter on the deployed application.'),
        cn.HEALTH_CHECK_KEYS.DEPLOYED_ON: openapi.String('Start up time of the server. Format YYYY-MM-DDTHH:MM:SS.ffffff, e.g. 2020-04-01T19:00:31.518627'),
        cn.HEALTH_CHECK_KEYS.APP_META: app_meta,
        cn.HEALTH_CHECK_KEYS.SERVICES: openapi.Object(
            'All available services on the server',
            additional_properties_type=openapi.Object(
                properties={
                    cn.HEALTH_CHECK_SERVICES_KEYS.ENDPOINT: openapi.String('Endpoint the service is exposed on.'),
                    cn.HEALTH_CHECK_SERVICES_KEYS.STATUS: openapi.String('Status of the model. If the app is ready the value will be "READY" .'
                                             'Otherwise the value will be a string indicating the status of the service.'),
                    cn.HEALTH_CHECK_SERVICES_KEYS.MODEL_CONTEXT: model_context
                }
            )
        )
    }
)


error_body = openapi.Object(
    properties=_error_body,
    reference_name='ErrorBody'
)


# TODO: just use one error object for all errors with model_context possibly empty?
# https://github.com/CadentTech/porter/issues/31

generic_error = openapi.Object(
    properties={
        **_base_response,
        cn.GENERIC_ERROR_KEYS.ERROR: error_body,
    },
    reference_name='GenericError'
)


model_context_error = openapi.Object(
    properties={
        **_base_response,
        cn.GENERIC_ERROR_KEYS.ERROR: error_body,
        cn.MODEL_CONTEXT_ERROR_KEYS.MODEL_CONTEXT: model_context
    },
    reference_name='ModelContextError'
)
