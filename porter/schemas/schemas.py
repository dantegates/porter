from . import openapi


request_id = openapi.String('Hex value of UUID assigned to the request.', reference_name='RequestID')


model_context = openapi.Object(
    properties={
        'model_name': openapi.String('The name of the model.'),
        'api_version': openapi.String('The model API version.'),
        'model_meta': openapi.Object(additional_properties_type=openapi.String('Arbitrary meta-data associated with the model.'))
    },
    reference_name='ModelContext'
)


health_check = openapi.Object(
    'Description of the applications status. Useful for load balancing and debugging',
    properties={
        'request_id': request_id,
        'porter_version': openapi.String('The version of the porter on the deployed application.'),
        'deployed_on': openapi.String('Start up time of the server. Format YYYY-MM-DDTHH:MM:SS.ffffff, e.g. 2020-04-01T19:00:31.518627'),
        'app_meta': openapi.Object(additional_properties_type=openapi.String('Arbitrary meta-data associated with the application')),
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
    properties={
        'messages': openapi.Array('An array of messages describing the error.', item_type=openapi.String()),
        'name': openapi.String('Name of the error')
    },
    reference_name='ErrorBody'
)


# TODO: just use one error object for all errors with model_context possibly empty?

generic_error = openapi.Object(
    properties={
        'request_id': request_id,
        'error': error_body
    }
)


model_context_error = openapi.Object(
    properties={
        'request_id': request_id,
        'error': error_body,
        'model_context': model_context
    },
    reference_name='ModelContextError'
)
