
Installation
============

``porter`` can be installed with ``pip`` for ``python3.9`` and higher as follows:

.. code-block:: shell

    pip install porter-schmorter  # because porter was taken


Version Selection
-----------------

If you want to install ``porter`` from a specific version, simply add ``==<version>``:

.. code-block:: shell

    pip install porter-schmorter==<version>


Dependency Selection
--------------------

``porter`` offers optional support for loading ``sklearn`` and ``keras`` models.  To enable each of these options explicitly:

.. code-block:: shell

    pip install -e git+https://github.com/dantegates/porter#egg=porter[sklearn-utils,keras-utils]

You can install just one of these additional requirements by removing the undesired name from the list in the brackets above (or you can install without optional dependencies by removing the bracketed list altogether).
