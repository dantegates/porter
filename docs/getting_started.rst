.. _getting_started:

Getting Started
===============

Getting started is as easy as:

.. code-block:: python

    from porter.datascience import WrappedModel
    from porter.services import ModelApp, PredictionService

    my_model = WrappedModel.from_file('my-model.pkl')
    prediction_service = PredictionService(
        model=my_model,
        name='my-model',
        api_version='v1')

    app = ModelApp([prediction_service])
    app.run()

Now just send a POST request to the endpoint ``/my-model/v1/prediction`` to get a prediction. Behind the scenes (with ``porter``'s default settings) your POST data will be converted to a ``pandas.DataFrame`` and the result of ``my_model.predict()`` will be returned to the user in a payload like the one below

.. code-block:: javascript

    {
        "model_context": {
            "api_version": "v1",
            "model_meta": {},
            "model_name": "my-model"
        },
        "predictions": [
            {
                "id": 1,
                "prediction": 0
            }
        ],
        "request_id": "0f86644edee546ee9c495a9a71b0746c"
    }

The model can be any Python object with a ``.predict(X)`` method, where ``X`` is a ``DataFrame`` and the return value is a sequence with one element per row of ``X``.  :meth:`WrappedModel.from_file` supports ``.pkl`` files via `joblib <https://joblib.readthedocs.io/>`_ and ``.h5`` files for `keras <https://keras.io/backend/>`_ models.

Multiple models can be served by a single app simply by passing additional services to :class:`porter.services.ModelApp`.

``porter`` also takes care of a lot of the boilerplate for you such as error handling. For example, by default, if the POST data sent to the prediction endpoint can't be parsed the user will receive a response with a 400 status code a payload describing the error.

.. code-block:: javascript

    {
        "error": {
            "messages": [
                "The browser (or proxy) sent a request that this server could not understand."
            ],
            "name": "BadRequest"
        },
        "model_context": {
            "api_version": "v1",
            "model_meta": {},
            "model_name": "my-model"
        },
        "request_id": "852ca09d578b447aa3d41d70b8cc4431"
    }

