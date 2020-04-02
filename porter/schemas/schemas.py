from . import robustify as rb


request_id = rb.String('Hex value of UUID assigned to the request.', reference_name='RequestID')


model_context = rb.Object(
    properties={
        'model_name': rb.String('The name of the model.'),
        'api_version': rb.String('The model API version.'),
        'model_meta': rb.Object(additional_params={'additionalProperties': rb.String('Arbitrary meta-data associated with the model.').to_openapi()})
    },
    reference_name='ModelContext'
)



health_check = rb.Object(
    'Description of the applications status. Useful for load balancing and debugging',
    properties={
        'request_id': request_id,
        'porter_version': rb.String('The version of the porter on the deployed application.'),
        'deployed_on': rb.String('Start up time of the server. Format YYYY-MM-DDTHH:MM:SS.ffffff, e.g. 2020-04-01T19:00:31.518627'),
        'app_meta': rb.Object(additional_params={'additionalProperties': rb.String('Arbitrary meta-data associated with the application').to_openapi()}),
        'services': rb.Object(
            'All available services on the server',
            additional_params={
                'additionalProperties': rb.Object(
                    properties={
                        'endpoint': rb.String('Endpoint the service is exposed on.'),
                        'status': rb.String('Status of the model. If the app is ready the value will be "READY" .'
                                            'Otherwise the value will be a string indicating the status of the service.'),
                        'model_context': model_context
                    }
                ).to_openapi()
            }
        )
    }
)


error_body = rb.Object(
    properties={
        'messages': rb.Array('An array of messages describing the error.', item_type=rb.String()),
        'name': rb.String('Name of the error')
    },
    reference_name='ErrorBody'
)


generic_error = rb.Object(
    properties={
        'request_id': request_id,
        'error': error_body
    }
)


model_context_error = rb.Object(
    properties={
        'request_id': request_id,
        'error': error_body,
        'model_context': model_context
    }
)