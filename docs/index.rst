.. porter documentation master file, created by
   sphinx-quickstart on Wed Feb 12 13:07:35 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

porter documentation
====================

`porter <https://github.com/CadentTech/porter/>`_ is a framework for exposing machine learning models via REST APIs.  Any object with a ``.predict()`` method will do which means ``porter`` plays nicely with models you have already trained using machine learning libraries such as `sklearn <https://scikit-learn.org/stable/>`_, `keras <https://keras.io/backend/>`_, or `xgboost <https://xgboost.readthedocs.io/en/latest/>`_.

It also includes the ability to load ``.pkl`` and ``.h5`` files so you don't have to write this code every time you deploy a new model and allows you to easily expose custom models.


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
