
Installation
============

``porter`` can be installed with ``pip`` for ``python3.6`` and higher as follows:

.. code-block:: shell

    pip install -e git+https://github.com/CadentTech/porter#egg=porter[all]

Note that without the ``-e`` flag and ``#egg=porter`` on the end of the url ``pip freeze`` will output ``porter==<version>`` rather than ``-e git+https://...`` as typically desired.


Version Selection
-----------------

If you want to install ``porter`` from a specific commit or tag, e.g. tag ``1.0.0`` simply add ``@<commit-or-tag>`` immediately before ``#egg=porter``:

.. code-block:: shell

    pip install -e git+https://github.com/CadentTech/porter@1.0.0#egg=porter

For more details on this topic see `here <https://codeinthehole.com/tips/using-pip-and-requirementstxt-to-install-from-the-head-of-a-github-branch/>`_.


Dependency Selection
--------------------

``porter`` offers optional support for loading ``keras`` models and for loading models via ``S3``.  To enable each of these options explicitly:

.. code-block:: shell

    pip install -e git+https://github.com/CadentTech/porter#egg=porter[keras-utils,s3-utils]

You can install just one of these additional requirements by removing the undesired name from the list in the brackets above (or you can install without optional dependencies by removing the bracketed list altogether).
