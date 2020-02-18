.. _tests:

Tests
=====

To run the test suite for porter execute the command

.. code-block:: shell

    make test

Additionally you can install a `git` pre-commit hook to run the test suite each time you make a commit with

.. code-block:: shell

    ./pre-commit-hook install

Note that you will need to set the following environment variables in order to run the AWS-related tests:

.. code-block:: shell

    export PORTER_S3_ACCESS_KEY_ID=[AWS access key id]
    export PORTER_S3_SECRET_ACCESS_KEY=[AWS secret access key]
    export PORTER_S3_BUCKET_TEST=[S3 bucket with rw access]
