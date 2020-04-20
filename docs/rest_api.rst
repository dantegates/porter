.. _rest_api:

REST API
========

Quick Guide
-----------

Two healtch check endpoints are exposed by each porter app: ``/-/alive`` and ``/-/ready``. These are useful for deploying a porter app in an environment like Kubernetes or behind a load balancer.

Additionally there is a prediction endpoint for each model service added to the :class:`porter.services.ModelApp` instance. The endpoint is computed from the name and version attributes of the model services: ``/<model name>/<model version>/prediction``. For example:

.. code-block:: python

    service1 = PredictionService(name='foo', version='v1', ...)
    service2 = PredictionService(name='bar', version='v2', ...)
    model_app.add_services(service1, service2)

will expose two models on the endpoints ``/foo/v1/prediction`` and ``/foo/v2/prediction``.  The endpoints accept POST requests with JSON payloads.


Defining API Schemas
--------------------

`porter` includes the ability to define API schemas for your services with explicity support
for the `OpenAPI <https://swagger.io/docs/specification/about/>` standard.

While this functionaity is completely optional it is particularly useful
providing the ability to automatically generated documentation and validate
request data. Additionally it can be used to generate an OpenAPI spec from
the command line which can be used for integration with the `vast toolset <https://openapi.tools/>` built around this standard.

Here's an example

.. literalinclude:: ../examples/contracts.py

    