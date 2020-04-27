.. _rest_api:

REST API
========

Below are human-friendly descriptions of the REST interface exposed by ``porter`` apps. Concrete descriptions of the API can be generated passing ``expose_docs=True`` to :class:`porter.services.ModelApp()`. See the :ref:`OpenAPI Schemas <schema_documentation>` page for more details.

.. _predictionservice_endpoints:

Prediction Service Endpoints
----------------------------

Each prediction service added to an instance of :class:`porter.services.ModelApp` (often referred to as a "porter app") is routed to its own endpoint.  The endpoint is computed from attributes of the model services: ``/<model name>/<model version>/prediction``.  For example:

.. code-block:: python

    service1 = PredictionService(name='foo', version='v1', ...)
    service2 = PredictionService(name='bar', version='v2', ...)
    model_app = ModelApp([service1, service2])

will expose two models on the endpoints ``/foo/v1/prediction`` and ``/ns/bar/v2/prediction``.  The endpoints accept POST requests with JSON payloads and return JSON payloads to the user.  POST responses will look something like:

.. code-block:: javascript

    {
        "model_context": {
            "api_version": "v1",
            "model_meta": {},
            "model_name": "foo"
        },
        "predictions": [
            {
                "id": 1,
                "prediction": 0
            }
        ],
        "request_id": "0f86644edee546ee9c495a9a71b0746c"
    }

For debugging purposes, the endpoints also accept GET requests, which simply return the message ``"This endpoint is live.  Send POST requests for predictions."``

Endpoint customization
^^^^^^^^^^^^^^^^^^^^^^

:class:`porter.services.PredictionService` endpoints can be customized with the ``namespace`` and ``action`` arguments.  For example,

.. code-block:: python

    service3 = PredictionService(
        name='baz', version='v1',
        namespace='datascience', action='pred', ...)
    model_app = ModelApp([service3])

results in the endpoint ``/datascience/baz/v1/pred``.

:ref:`Custom services <baseservice>` also have the flexibility to define their endpoints.


.. _health_checks:

Health Checks
-------------

Two health check endpoints are exposed by each ``porter`` app (not for each service): ``/-/alive`` and ``/-/ready``.  These are useful for load balancing a ``porter`` app.  Each health check endpoint returns a JSON payload with metadata about the deployment and services that are running, e.g.:

.. code-block:: javascript

    {
      "app_meta": {},
      "deployed_on": "2020-04-01T12:00:00.445124",
      "porter_version": "0.15.0",
      "request_id": "e59b0ab32fe94ea1a31cb289a36baf51",
      "services": {
        "/my-model/v1/prediction": {
          "endpoint": "/my-model/v1/prediction",
          "model_context": {
            "api_version": "v1",
            "model_meta": {},
            "model_name": "my-model"
          },
          "status": "READY"
        },
      }
    }

If the app is running, the ``/-/alive`` endpoint response will have a 200 status code. The ``/-/ready`` endpoint will return a 503 if any of the services added to the :class:`porter.services.ModelApp` indicate that they are not ready.

.. note::

    Although all services included in ``porter`` are always considered ready, distinguishing between "liveness" and "readiness" is expected by many platforms `such as Kubernetes <https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/>`_. Exposing both now allows us to support services that may make that distinction in the future without users having to change their code.

Error Objects
-------------

Responses to requests that result in client or server side errors will return the appropriate status code and a payload with information describing the error and request context. Such payloads contain ``error`` and ``model_context`` objects as well as the ``request_id``.

.. code-block:: json

    {
        "error": {
            "messages": [
                "Schema validation failed: data must be array"
            ],
            "name": "UnprocessableEntity"
        },
        "model_context": {
            "api_version": "v2",
            "model_meta": {},
            "model_name": "user-ratings"
        },
        "request_id": "e7fd6560f6614a77bd762f878ea1dd7f"
    }



Status Codes
^^^^^^^^^^^^

Clients should be prepared to handle the following error codes from service endpoints.

- **400**: Bad Request. Raised when the payload cannot be parsed.
- **422**: Unprocessable Entity. This also indicates there is an error in the request payload, but raises the distinction that although the data was valid JSON, it contains semantic errors. This includes invalid schemas or user raised errors (from ``check_request``).
- **500**: Something went wrong when ``model.predict`` was called.
