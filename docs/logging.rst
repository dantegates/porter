
Logging
=======


API calls (including request and response payloads) can be logged by passing ``log_api_calls=True`` when instantiating a service. The following data is logged at the ``'INFO'`` level

- ``"request_id"``: A unique ID for the request.
- ``"request_data"``: The request's JSON payload.
- ``"response_data"``: The response's JSON payload.
- ``"service_class"``: The name of the service class that served the request, e.g. ``PredictionService``.
- ``"event"``: The type of event being logged, e.g. "request" or "response".

By default, the ``porter`` logger sends logs to a null handler. To actually obtain the logs you must add
a handler to the ``porter`` logger and formatter with the desired logging attributes.

For example

.. code-block:: python

    from logging import getLogger, Formatter, StreamHandler
    logger = getLogger('porter')
    logger.setLevel('INFO')
    logger.setFormatter(Formatter('%(asctime)-15s %(request_id)s %(event)-8s %(message)s')
    logger.addHandler(StreamHandler())

Notice that in the example above the formatter uses default Python log attributes as well as
custom values available from ``porter`` such as "request_id".

JSON Formatter
--------------

For convenience ``porter`` includes a log formatter that converts log records to JSON.
This is particularly useful for allowing users to programmatically query logs emitted
from ``porter``.

The JSON log formatter can be set as follows:

.. code-block:: python

    from porter.utils import JSONLogFormatter
    logger = getLogger('porter')
    formatter = JSONLogFormatter(
        'asctime', 'levelname', 'module', 'name', 'message',
        'request_id', 'service_class', 'event')
    logger.setFormatter(formatter)

which generates logs (pretty printed for this example only) such as:

.. code-block:: json

    {
        "levelname": "INFO",
        "service_class": "PredictionService",
        "request_id": "6c26423fe85445948071d01283aaf58e",
        "event": "api_call",
        "message": "api logging",
        "asctime": "2020-04-21 07:02:05,569",
        "module": "services",
        "name": "porter.services"
    }


A full working example can be found in the example script `examples/api_logging.py <https://github.com/dantegates/porter/blob/master/examples/api_logging.py>`_.

User Logging
------------

In some cases users may want to add the ``porter`` ``request_id`` to their logs
to associate their application logs with ``porter``'s default logs. This can
be accomplished with :meth:`porter.api.request_id`. Note that this function should
only be called while an active request is being handled.

.. code-block:: python

    from porter.api import request_id

    # code called while a request is being processed
    logger.info('the first message', extra={'request_id': request_id())
    logger.info('the second message', extra={'request_id': request_id())

In the example above, if a `log formatter <https://docs.python.org/3/library/logging.html#formatter-objects>`_
using the format ``'[request_id %(request_id)] %(msg)s]'`` was added to ``logger``,
the code above would generate the following logs for a single request.

.. code-block:: text

    [request_id 6c26423fe85445948071d01283aaf58e] the first message
    [request_id 6c26423fe85445948071d01283aaf58e] the second message
