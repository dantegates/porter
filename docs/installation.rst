.. _installation:

Installation
============

``porter`` can be installed with ``pip`` as follows

.. code-block:: shell

    pip install -e git+https://github.com/CadentTech/porter#egg=porter

Note that without the ``-e`` flag and ``#egg=porter`` on the end of the url ``pip freeze`` will output ``porter==<version>``
rather than ``-e git+https://...`` as typically desired.

If you want to install ``porter`` from a specific commit or tag, e.g. tag ``1.0.0`` simply add
``@<commit-or-tag>`` immediately before ``#egg=porter``.

.. code-block:: shell

    pip install -e git+https://github.com/CadentTech/porter@1.0.0#egg=porter

For more details on this topic see `here <https://codeinthehole.com/tips/using-pip-and-requirementstxt-to-install-from-the-head-of-a-github-branch/>`_.

Installing extra dependencies
-----------------------------

Porter supports functionality for loading ``sklearn`` and ``keras`` models either from disk or ``S3``.
Since we don't expect ``porter`` applications to necessarily need all of these dependencies at once
they can be optionally installed by issuing the command from the root directory of this repository

.. code-block:: shell

    pip install -e git+https://github.com/CadentTech/porter#egg=porter[keras-utils,sklearn-utils,s3-utils]

Note that you can choose to install only a subset of these additional requirements by removing
the undesired names from the comma separated list in brackets above.

