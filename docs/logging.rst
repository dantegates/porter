.. _logging:

Logging
=======


API calls (request and response payloads) can be logged by passing ``log_api_calls=True`` when instantiating a service. The following data is available for logging

- ``"request_id"``: A unique ID for the request.
- ``"request_data"``: The request's JSON payload.
- ``"response_data"``: The response's JSON payload.
- ``"service_class"``: The name of the service class that served the request.
- ``"event"``: The type of event being logged, e.g. "request" or "response".

The script `examples/api_logging.py <https://github.com/CadentTech/porter/blob/master/examples/api_logging.py>`_ demonstrates how to configure logging.  

.. todo::
    Add info on additional loggable values.
