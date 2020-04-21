.. _rest_api:

REST API
========


Prediction Endpoints
--------------------

There is a prediction endpoint for each model service added to the :class:`porter.services.ModelApp` instance.  The endpoint is computed from the name and version attributes of the model services: ``/<model name>/<model version>/prediction``.  For example:

.. code-block:: python

    service1 = PredictionService(name='foo', version='v1', ...)
    service2 = PredictionService(name='bar', version='v2', ...)
    model_app = ModelApp([service1, service2])

will expose two models on the endpoints ``/foo/v1/prediction`` and ``/foo/v2/prediction``.  The endpoints accept POST requests with JSON payloads and return JSON payloads to the user.  For debugging purposes, the endpoints also accept GET requests, which return the message "This endpoint is live.  Send POST requests for predictions."

Endpoint customization
^^^^^^^^^^^^^^^^^^^^^^

:class:`porter.services.PredictionService` allows a custom endpoint URL through the ``namespace`` and ``action`` arguments.  For example,

.. code-block:: python

    service3 = PredictionService(
        name='baz', version='v1',
        namespace='datascience', action='pred', ...)
    model_app = ModelApp([service3])

results in a prediction endpoint ``/datascience/baz/v1/pred``.


Health Checks
-------------

Two health check endpoints are exposed by each ``porter`` app (not for each service): ``/-/alive`` and ``/-/ready``.  These are useful for deploying a ``porter`` app in an environment like Kubernetes or behind a load balancer.  Each health check endpoint returns a JSON payload with metadata about the deployment and services that are running, e.g.:

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


