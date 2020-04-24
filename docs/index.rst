.. porter documentation master file, created by
   sphinx-quickstart on Wed Feb 12 13:07:35 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

porter documentation
====================

`porter <https://github.com/CadentTech/porter/>`_ is a framework for data scientists who want to quickly and reliably deploy machine learning models as REST APIs. 

Simplicity is a core goal of this project. The following 6 lines of code are a fully functional example. While this should the most common use case, ``porter`` is also designed to be easily extended to cover the remaining cases not supported out of the box.

.. code-block::

    from porter.datascience import WrappedModel
    from porter.services import ModelApp, PredictionService

    my_model = WrappedModel.from_file('my-model.pkl')
    prediction_service = PredictionService(model=my_model, name='my-model', api_version='v1')

    app = ModelApp([prediction_service])
    app.run()


Features include:

- **Practical design**: suitable for projects ranging from proof-of-concept to production grade software.
- **Framework-agnostic design**: any object with a ``predict()`` method will do, which means ``porter`` plays nicely with `sklearn <https://scikit-learn.org/stable/>`_, `keras <https://keras.io/backend/>`_, or `xgboost <https://xgboost.readthedocs.io/en/latest/>`_ models. Models that don't fit this pattern can be easily wrapped and used in ``porter``.
- **OpenAPI integration**: lightweight, Pythonic schema specifications support automatic validation of HTTP request data and generation of API documentation using Swagger.
- **Boiler plate reduction**: ``porter`` takes care of API logging and error handling out of the box, and supports streamlined model loading from ``.pkl`` and ``.h5`` files stored locally or on AWS S3.
- **Robust testing**: a comprehensive test suite ensures that you can use ``porter`` with confidence. Additionally, ``porter`` has been extensively field tested by the Data Science team at Cadent.


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation
   getting_started
   rest_api
   openapi_schemas
   deployment
   custom_models
   error_handling
   logging
   unit_testing
   contributing
   examples
   porter

* :ref:`genindex`
* :ref:`search`


.. check out this crazy vim macro that helped in doubling up on backticks:
.. ls`lf`.ll/[^:)]`\w

.. vim: tw=0
