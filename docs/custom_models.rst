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

.. todo::
    Borrow from contracts.py


.. _fullycustom:

Fully Customized Models
-----------------------

.. todo::
    Borrow from contracts.py
