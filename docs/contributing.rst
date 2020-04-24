
Contributing
============

Running porter's test suite
---------------------------

To run the test suite for ``porter``, execute the command:

.. code-block:: shell

    make test

Additionally you can install a ``git`` pre-commit hook to run the test suite each time you make a commit with:

.. code-block:: shell

    ./pre-commit-hook install


Note that `test_loading.py <https://github.com/CadentTech/porter/blob/master/tests/test_loading.py>`_ requires the following environment variables in order to test ``porter``'s AWS S3 loading capabilities:

.. code-block:: shell

    export PORTER_S3_ACCESS_KEY_ID=[AWS access key id]
    export PORTER_S3_SECRET_ACCESS_KEY=[AWS secret access key]
    export PORTER_S3_BUCKET_TEST=[S3 bucket with rw access]

This of course also requires the ``s3-utils`` optional dependency to be enabled.


Submitting code
---------------

If you would like to share enhancements to ``porter`` code or documentation, `fork us on GitHub <https://github.com/CadentTech/porter>`_ and make your changes in a new branch; then `submit a pull request <https://github.com/CadentTech/porter/pulls>`_ for review.

For any contributions, we ask that any new behavior be validated by the test suite, which must be passing in its entirety.


Reaching out
------------

To report a bug or request a new feature, describe your situation in the `issue tracker <https://github.com/CadentTech/porter/issues>`_ and use the *bug* or *enhancement* label as appropriate.
