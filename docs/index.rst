.. porter documentation master file, created by
   sphinx-quickstart on Wed Feb 12 13:07:35 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

porter documentation
====================

`porter <https://github.com/CadentTech/porter/>`_ is a framework for  data scientists who want to quickly and reliably deploy machine learning models as REST APIs. 


In particular ``porter``

- Is Designed to be a practical solution for projects from POCs to production grade software.
- Doesn't make assumptions about what frameworks you should use. Any object with a ``predict()`` method will do which means ``porter`` plays nicely with `sklearn <https://scikit-learn.org/stable/>`_, `keras <https://keras.io/backend/>`_, or `xgboost <https://xgboost.readthedocs.io/en/latest/>`_ models. Models that don't fit this pattern can be easily wrapped and used in porter.
- Integrates OpenAPI support for validating HTTP Request data and automatically generating API documentation with Swagger.
- Implements API logging and error handling out of the box.
- Is built on a robust test suite so you can use ``porter`` with confidence. Additionally, ``porter`` has been extensively field tested at Cadent by our Data Science team.
- Handles boiler plate like such as loading ``.pkl`` and ``.h5`` files.


Simplicity is also a core goal of this project. The following 6 lines of code are a fully functional example. While this should the most common use cases, ``porter`` is also designed to be easily extended to cover the remaining cases not supported out of the box.


.. code-block::

   from porter.datascience import WrappedModel
   from porter.services import ModelApp, PredictionService

   my_model = WrappedModel.from_file('my-model.pkl')
   prediction_service = PilotPredictionService(model=my_model, name='my-model', api_version='v1')

   app = ModelApp([prediction_service])
   app.run()

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
