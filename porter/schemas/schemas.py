from .. import config as cf
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
        'model_name': openapi.String('The name of the model.'),
        'api_version': openapi.String('The model API version.'),
        'model_meta': model_meta
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
        'porter_version': openapi.String('The version of the porter on the deployed application.'),
        'deployed_on': openapi.String('Start up time of the server. Format YYYY-MM-DDTHH:MM:SS.ffffff, e.g. 2020-04-01T19:00:31.518627'),
        'app_meta': app_meta,
        'services': openapi.Object(
            'All available services on the server',
            additional_properties_type=openapi.Object(
                properties={
                    'endpoint': openapi.String('Endpoint the service is exposed on.'),
                    'status': openapi.String('Status of the model. If the app is ready the value will be "READY" .'
                                             'Otherwise the value will be a string indicating the status of the service.'),
                    'model_context': model_context
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
        'error': error_body,
    },
    reference_name='GenericError'
)


model_context_error = openapi.Object(
    properties={
        **_base_response,
        'error': error_body,
        'model_context': model_context
    },
    reference_name='ModelContextError'
)
