.. _service_architecture:

Service Architecture
====================

``porter`` provides a micro-service architecture, in which an *app* routes traffic to one or more *services* with well-defined, minimal interfaces.  The *app* is an instance of :class:`porter.services.ModelApp`, and *services* are instances of classes, such as :class:`porter.services.PredictionService`, that are derived from :class:`porter.services.BaseService`.  We've outlined basic usage of these classes in previous pages; here we discuss each of them in greater detail.


ModelApp
--------

:class:`ModelApp <porter.services.ModelApp>` is responsible for providing an interface to each registered service, health checks, and optionally documentation.  The trivial app ``ModelApp([])`` just exposes the health check endpoints ``/-/alive`` and ``/-/ready``.  An app that engages all available functionality might look like this:

.. code-block:: python

    from porter.services import ModelApp

    app = ModelApp(
        [service1, service2, ...],
        name='Busy App',
        description="""
        <p>An app that exposes plenty of services.</p>
        <p><b>Created by</b>: Jack and Jill</p>
        """,
        description='An app that exposes plenty of services',
        version='1.0.37',
        meta={'creators': 'Jack and Jill', 'release-date': '2020-04-01'},
        expose_docs=True,
        docs_url='/documentation/',
        docs_json_url='/_documentation.json',
        docs_prefix='/models/busy_app/')

At present, all keyword arguments to :class:`ModelApp() <porter.services.ModelApp()>` are optional.  Here are
their effects:

- ``name``, ``description``, ``version``: These set the title, subtitle, and version badge at the top of the documentation.  The description can optionally make use of HTML tags.  The ``porter`` version will be appended to the description.
- ``meta``: This sets the ``app_meta`` object returned by the health checks (see :ref:`health_checks`).
- ``expose_docs``: This enables automatic documentation.
- ``docs_url``: This determines the URI where the documentation is hosted; by default this is ``/docs/``.  Note that GET requests to ``/`` forward to this URI.
- ``docs_json_url``: This determines the URI for a JSON representation of the `Swagger <https://swagger.io>`_ input; by default this is ``/_docs.json``.  This can be useful for interfacing with other `Swagger-related tools <https://swagger.io/tools/open-source/>`_.
- ``docs_prefix``: This locates the documentation somewhere other than the root level.  This is useful, for example, if the app will be deployed behind a load balancer.  In this example, suppose Busy App is hosted at ``[domain]/models/busy_app/``; configuring ``docs_prefix`` allows the documentation to be served accordingly from ``[domain]/models/busy_app/documentation/``.



PredictionService
-----------------

:class:`PredictionService <porter.services.PredictionService>` is the workhorse class for serving data science models.  In :ref:`getting_started`, we saw the minimal usage of :class:`PredictionService <porter.services.PredictionService>`:

.. code-block:: python

    from porter.services import PredictionService

    prediction_service = PredictionService(
        model=my_model,
        name='my-model',
        api_version='v1')

An instance that engages all available functionality might look like this:

.. code-block:: python

    prediction_service = PredictionService(
        model=model,
        name='supa-dupa-model',
        api_version='v1',
        meta={'creators': 'Alice & Bob'},
        log_api_calls=True,
        namespace='datascience',
        action='prediction',
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        batch_prediction=False,
        additional_checks=mychecks,
        feature_schema=feature_schema,
        prediction_schema=prediction_schema,
        validate_request_data=True,
        validate_response_data=True)

Here are the effects of the optional keyword arguments:

- ``meta``: This sets the ``model_meta`` object that is returned as part of the ``model_context`` in :ref:`POST responses <predictionservice_endpoints>`.
- ``log_api_calls``: This enables logging; see :ref:`logging`.
- ``namespace``, ``action``: These, along with ``name`` and ``api_version``, determine the prediction endpoint: ``/<namespace>/<name>/<api version>/<action>/``.
- ``preprocessor``, ``postprocessor``: These allow transformations to be made to the input and output, immediately before and after ``model.predict()``.  See :ref:`ex_example` and the :class:`PredictionService() <porter.services.PredictionService>` docstring for more details.
- ``batch_prediction``: See :ref:`instance_prediction` below.
- ``additional_checks``: Optional callable taking input DataFrame ``X`` and raising a ``ValueError`` for invalid input.  This is intended for input validation against complex constraints that cannot be expressed entirely using ``feature_schema``.
- ``feature_schema``, ``prediction_schema``, ``validate_request_data``, ``validate_response_data``: Input and output schemas for automatic validation and/or documentation.  See also :ref:`openapi_schemas` as well as :ref:`custom_prediction_schema` below.

.. _instance_prediction:

Instance Prediction
^^^^^^^^^^^^^^^^^^^

For models with expensive predictions, you may wish to enforce that prediction is run on individual instances at a time.  For this behavior, request ``batch_prediction=False``, e.g.:

.. code-block:: python

    prediction_service = PredictionService(
        model=my_model,
        name='my-model',
        api_version='v1',
        batch_prediction=False)

Now the model will accept input of the form of a single ``object``

.. code-block:: json

    {
        "id": 1,
        "user_id": 122333,
        "title_id": 444455555,
        "is_tv": true,
        "genre": "comedy",
        "average_rating": 6.7
    }

as opposed to the usual ``array``:


.. code-block:: json

    [
        {
            "id": 1,
            "user_id": 122333,
            "title_id": 444455555,
            "is_tv": true,
            "genre": "comedy",
            "average_rating": 6.7
        }
    ]

.. note::

    ``batch_prediction=False`` does not fundamentally change the way ``porter`` interacts with the underlying model object; it simply enforces that the input must include only a single object.  Internally, the input is still converted into a ``pandas.DataFrame`` with a single row.  For a model which fundamentally accepts only a single object as an input, see :ref:`baseservice`.

.. _custom_prediction_schema:

Custom Prediction Schema
^^^^^^^^^^^^^^^^^^^^^^^^

By default, :class:`PredictionService <porter.services.PredictionService>` assumes that each prediction is a single scalar value, which can be represented by the following simple schema:

.. code-block:: python

    default_prediction_schema = porter.schemas.Number('Model Prediction')

However, custom models may return more complex outputs.  For example, suppose we have a probabilistic model that returns lower and upper bounds in addition to an expected value.  Here is an example model definition that doesn't do anything but give us a working example:

.. code-block:: python

    import pandas as pd
    import scipy.stats as ss

    class ProbabilisticModel(BaseModel):
        def predict(self, X):
            dist = ss.norm(ss.norm(0, 1).rvs(len(X)), 1)
            return pd.DataFrame({
                'lower_bound': dist.ppf(0.05),
                'expected_value': dist.mean(),
                'upper_bound': dist.ppf(0.95),
            }).to_dict(orient='records')

The ``predict()`` method of this model accepts a ``DataFrame`` and returns a list of dictionaries, one per input row.  Output of this form is sufficient for yielding valid response JSON payloads with non-scalar predictions.

For `automatically generating <openapi_schemas.html#schema-documentation>`_ appropriate documentation for such a model, the per-row prediction schema could be described as:

.. code-block:: python

    proba_ratings_prediction_schema = Object(
        'Return a prediction with upper and lower bounds',
        properties={
            'lower_bound': Number(
                'Lower bound on the prediction. '
                'Actual values should fall below this range just 5% of the time'),
            'expected_value': Number(
                'The average value we expect actual values to take.'),
            'upper_bound': Number(
                'Upper bound on the prediction. '
                'Actual values should fall above this range just 95% of the time'),
        },
        reference_name='ProbaModelPrediction')

And the prediction service could be instantiated as:

.. code-block:: python

    probabilistic_service = PredictionService(
        model=ProbabilisticRatingsModel(),
        name='proba-model',
        api_version='v1',
        feature_schema=ratings_feature_schema,
        prediction_schema=proba_ratings_prediction_schema)

In your own tests of ``probabilistic_service``, you can validate the response data by:

.. code-block:: python

    probabilistic_service.response_schema.validate(response)

.. warning::

    There is also experimental support for automatic response validation: ``PredictionService(..., validate_response_data=True)``.  Enabling this feature triggers a warning stating that it may increase response latency and produce confusing error messages for users.  This should only be used for testing/debugging.


.. _baseservice:

Subclassing BaseService
-----------------------

By subclassing :class:`BaseService <porter.services.BaseService>` it is possible to expose arbitrary Python code.  Consider complex input and output schemas such as:

.. code-block:: python

    from porter.schemas import Object, Array, String, Integer

    custom_service_input = Object(
        properties={
            'string_with_enum_prop': String(additional_params={'enum': ['a', 'b', 'abc']}),
            'an_array': Array(item_type=Number()),
            'another_property': Object(properties={'a': String(), 'b': Integer()}),
            'yet_another_property': Array(item_type=Object(additional_properties_type=String()))
        },
        reference_name='CustomServiceInputs'
    )

    custom_service_output_success = Object(
        properties={
            'request_id': request_id,
            'model_context': model_context,
            'results': Array(item_type=String())
        }
    )

A minimal app implementing and documenting this interface might look like:

.. code-block:: python

    from porter.services import BaseService, ModelApp

    class CustomService(BaseService):
        # action: required property or class attribute
        action = 'custom-action'

        # route_kwargs: required property or class attribute
        route_kwargs = {'methods': ['POST']}

        # status: required property
        @property
        def status(self):
            return 'READY'

        # serve: required method taking no arguments
        def serve(self):
            data = self.get_post_data()
            return {'results': ['foo', 'bar']}

    custom_service = CustomService(
        name='custom-service',
        api_version='v1',
        validate_request_data=True)
    custom_service.add_request_schema('POST', custom_service_input)
    custom_service.add_response_schema('POST', 200, custom_service_output_success)
    custom_app = ModelApp([custom_service], expose_docs=True)

This would expose an endpoint ``/custom-service/v1/custom-action``.

.. note::
    Unlike ``PredictionService``, custom subclasses of ``BaseService`` will receive POST data and deliver response data directly, with no automatic conversion to ``pandas.DataFrame``.

For a more complex example that serves calculations from a callable function, more closely matching the behavior of :class:`PredictionService <porter.services.PredictionService>`, see the :ref:`ex_function_service` example script.

