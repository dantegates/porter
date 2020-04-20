.. _custom_models:

Custom Models
=============

In addition to exposing standard ``sklearn``-compatible models, ``porter`` supports three styles of customization: instance prediction, custom prediction schemas, and fully customized model types.


Instance Prediction
-------------------

For models with expensive predictions, you may wish to enforce that prediction is run on individual instances at a time.  For this behavior, simply request ``batch_prediction=False``, e.g.:

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

    ``batch_prediction=False`` does not fundamentally change the way ``porter`` interacts with the underlying model object; it simply enforces that the input must include only a single object.  Internally, the input is still converted into a ``pandas.DataFrame`` with a single row.  For a model which fundamentally accepts only a single object as an input, see `Fully Customized Models <fullycustom_>`_.


Custom Prediction Schema
------------------------

Suppose we have a probabilistic model that returns more than a single scalar value for each prediction.  Here is an example model definition that doesn't do anything but give us a working example:

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

.. warning::

    There is also experimental support for response validation: ``PredictionService(..., validate_response_data=True)``.  Enabling this feature triggers a warning stating that it may increase response latency and produce confusing error messages for users.  This should only be used for testing/debugging.


.. _fullycustom:

Fully Customized Models
-----------------------

By subclassing :class:`porter.services.BaseService` it is possible to expose arbitrary Python code.

.. note::
    We have sometimes found it useful to subclass ``BaseService``.  However, this usage depends on implementation details that may change in future releases.

Consider complex input and output schemas such as:

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
        action = 'custom-action'
        route_kwargs = {'methods': ['POST']}

        def serve(self):
            data = self.get_post_data()
            return {'results': ['foo', 'bar']}

        @property
        def status(self):
            return 'READY'

    custom_service = CustomService(
        name='custom-service',
        api_version='v1',
        validate_request_data=True)
    custom_service.add_request_schema('POST', custom_service_input)
    custom_service.add_response_schema('POST', 200, custom_service_output_success)
    custom_app = ModelApp([custom_service], expose_docs=True)

This would expose an endpoint ``/custom-service/v1/custom-action``.

Here is a more complex example that serves calculations from a callable function:

.. code-block:: python

    import porter.api as porter_api

    class FunctionService(BaseService):

        route_kwargs = {'methods': ['GET', 'POST'], 'strict_slashes': False}

        def __init__(self, action, function,
                     input_schema=None,
                     output_schema=None,
                     additional_checks=None,
                     **kwargs):
            self._action = action
            super().__init__(**kwargs)
            if not callable(function):
                raise ValueError('`function` must be callable')
            self.callable = function
            if input_schema is not None:
                self.add_request_schema('POST', input_schema)
            self.add_response_schema('GET', 200, sc.String())
            if output_schema is not None:
                self.add_response_schema('POST', 200, output_schema)
            if additional_checks is not None and not callable(additional_checks):
                raise ValueError('`additional_checks` must be callable')
            self.additional_checks = additional_checks

        @property
        def action(self):
            return self._action

        @property
        def status(self):
            return 'READY'

        def serve(self):
            if porter_api.request_method() == 'GET':
                return f"This endpoint is live. Send POST requests for '{self.action}'."
            data = self.get_post_data()
            if self.additional_checks is not None:
                self.additional_checks(data)
            out = self.callable(data)
            return out

This could be used, for example, to expose some NumPy functions:

.. code-block:: python

    from porter.exceptions import PorterException
    import numpy as np

    def sum(x):
        return np.sum(x).tolist()

    def prod(x):
        return np.prod(x).tolist()

    def check_for_zeros(x):
        if 0 in x:
            raise PorterException('input cannot include zeros', code=422)

    input_schema = sc.Array(item_type=sc.Number(), reference_name='InputSchema')
    output_schema = sc.Number(reference_name='OutputSchema')
    service_kw = dict(
        input_schema=input_schema,
        output_schema=output_schema,
        validate_request_data=True)

    sum_service = FunctionService('sum', sum, name='math', api_version='v1', **service_kw)
    prod_service = FunctionService('prod', prod, name='math', api_version='v1',
                                   additional_checks=check_for_zeros, **service_kw)

    app = ModelApp(
        [sum_service, prod_service],
        name='FunctionService Example',
        description='Expose arbitrary callable functions by subclassing BaseService.',
        expose_docs=True)


