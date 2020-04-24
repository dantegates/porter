.. _error_handling:

Error Handling
==============

``porter`` error handling is similar to ``flask``. When an exception goes
unhandled while processing an HTTP request, if the exception has the attribute
``code`` then ``porter`` will use this value as the status code for the HTTP
response.

``porter`` is designed to cover exception handling for 99% of use cases. However,
there are legitimate reasons users may need to implement custom exception
handling. For example, suppose you want to use ``porter`` to expose documentation
for a model REST API before the model is ready to start serving requests. In
this case, you might want to deploy a model whose endpoint returns 503. This can be
accomplished as follows

.. code-block:: python

    from porter.exceptions import PorterException
    from porter.services import PredictionService

    class StubService(PredictionService):
        def serve(self):
            raise PorterException('Model not yet available', code=503)
