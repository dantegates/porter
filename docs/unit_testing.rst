.. unittests:

Unit Testing porter Apps
========================

.. warning::

    Currently ``porter`` uses `flask <https://flask.palletsprojects.com/en/1.1.x/>`_ to implement a WSGI server. Even though we are about to recommend using the underlying ``flask`` app in your unittests, this should be considered an implementation detail elsewhere.

    *If* one day the decision is made no longer to use ``flask``, the developers of ``porter`` will have to rewrite *soooo many* unit tests in order to use the upgrades ourselves. Therefore you can rest assured that *if* this ever happens, we will make accommodations for easily updating test suites.

    That said, none of our production code is using ``@model_app.app.route(...)`` anywhere, so outside of subclasses of ``unittest.TestCase`` we recommend to forget all about this little secret.


Using a test client
-------------------

The following code demonstrates how to write unit tests for ``porter`` apps that make API calls against a mock server.


.. code:: python

    model_app = ModelApp(...)
    test_client = model_app.app.test_client()
    test_client.post(...)

As mentioned above, the test client is actually a `flask test client <https://flask.palletsprojects.com/en/1.1.x/api/#flask.Flask.test_client>`_. See the flask documentation for more details.
