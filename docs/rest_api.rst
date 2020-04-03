.. _rest_api:

REST API
========




Quick Guide
-----------

Two healtch check endpoints are exposed by each porter app: ``/-/alive`` and ``/-/ready``. These are useful for deploying a porter app in an environment like Kubernets or behind a load balancer.

Additionally there is a prediction endpoint for each model service added to the :class:`porter.services.ModelApp` instance. The endpoint is computed from the name and version attributes of the model services: ``/<model name>/<model version>/prediction``. For example:

.. code-block:: python

    service1 = PredictionService(name='foo', version='v1', ...)
    service2 = PredictionService(name='bar', version='v2', ...)
    model_app.add_services(service1, service2)

will expose two models on the endpoints ``/foo/v1/prediction`` and ``/foo/v2/prediction``.  The endpoints accept POST requests with JSON payloads.
