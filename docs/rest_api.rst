.. _rest_api:

REST API
========


The complete details of an API exposed by ``porter`` can be found in the `openapi <https://openapi.tools/>`_ spec in `this repository <https://github.com/CadentTech/porter/tree/master/openapi>`_.

Additionally you can open the `static documentation <https://github.com/CadentTech/porter/blob/master/openapi/html/index.html>`_ generated from this spec in your web browser. On a Mac:

.. code-block:: shell

    open openapi/html/index.html

For those who just want to get a sense of the API a quick overview is below.

Quick Guide
-----------

Two healtch check endpoints are exposed by each porter app: ``/-/alive`` and ``/-/ready``. These are useful for deploying a porter app in an environment like Kubernets or behind a load balancer.

Additionally there is a prediction endpoint for each model service added to the :class:`porter.services.ModelApp` instance. The endpoint is computed from the name and version attributes of the model services: ``/<model name>/<model version>/prediction``. For example:

.. code-block:: python

    service1 = PredictionService(name='foo', version='v1', ...)
    service2 = PredictionService(name='bar', version='v2', ...)
    model_app.add_services(service1, service2)

will expose two models on the endpoints ``/foo/v1/prediction`` and ``/foo/v2/prediction``.  The endpoints accept POST requests with JSON payloads.
