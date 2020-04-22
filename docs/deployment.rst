.. _deployment:

Deployment
==========

Production deployment
---------------------

The recommended way to use ``porter`` in production is through a production-grade WSGI server, such as  `gunicorn <https://gunicorn.org/>`_. To do so simply define an instance of :class:`porter.services.ModelApp` in your python script and then point gunicorn to it.

For example, given a python script ``app.py`` closing with:

.. code-block:: python

    model_app = ModelApp(...)

Then for production use, either in a shell script or on the command line, invoke:

.. code-block:: shell

    gunicorn app:model_app

For more options, see e.g. `deployment options <https://flask.palletsprojects.com/en/1.1.x/deploying/#deployment>`_ in the Flask documentation.



Local testing deployment
------------------------

For pre-production testing and debugging, it is appropriate to run the app's development server directly:

.. code-block:: python

    model_app = ModelApp(...)

    if __name__ == '__main__':
        model_app.run()

This usage will result in a warning to remind you to upgrade for production deployment:

.. code-block:: shell

    WARNING: This is a development server. Do not use it in a production deployment.
