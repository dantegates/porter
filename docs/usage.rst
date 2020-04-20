.. _usage:

Usage
=====

The basic workflow for building a model service is as follows

1. Instantiate an instance of :class:`porter.services.ModelApp`. This is simply an abstraction for the REST server that will expose your models.
2. Define model classes for each service you want to add to the app. These classes should must expose a ``.predict()`` method. Additionally you can define pre- and post-processor classes (objects implementing the :class:`porter.datascience.BasePreProcessor` or :class:`porter.datascience.BasePreProcessor` interface) for pre/post processing of model input/output respectively. If you have a serialized sklearn and/or keras object and/or your model is on S3, classes in :class:`porter.datascience` can help load these objects.
3. Instantiate classes, such as :class:`porter.services.PredictionService`, with the appropriate arguments for each model you would expose through the app.
4. Pass the service defined in (3) to the ``add_serivce()`` method of your :class:`porter.services.ModelApp` instance.
5. Call the ``run()`` method of your :class:`porter.services.ModelApp` instance. Your model is now live!

See this `examples/example.py <https://github.com/CadentTech/porter/blob/master/examples/example.py>`_ for an (almost functional) example.


Running porter apps in production
---------------------------------

There are two ways to run porter apps. The first is calling the :meth:`porter.services.ModelApp.run` method. This is just a wrapper to the underlying  `flask <https://flask.palletsprojects.com/>`_  app which is good for development but not for production. A better way to run porter apps in production is through a production-grade WSGI server, such as  `gunicorn <https://gunicorn.org/>`_. To do so simply define an instance of :class:`porter.services.ModelApp` in your python script and then point gunicorn to it.

For example, in your python script ``app.py``

.. code-block:: python

    model_app = ModelApp(...)

Then for production use, either in a shell script or on the command line

.. code-block:: shell

    gunicorn app:model_app

