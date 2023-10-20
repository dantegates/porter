
Contributing
============


Reaching out
------------

To report a bug or request a new feature, describe your situation in the `issue tracker <https://github.com/CadentTech/porter/issues>`_ and use the *bug* or *enhancement* label as appropriate.


Submitting code
---------------

If you would like to share enhancements to ``porter`` code or documentation, `fork us on GitHub <https://github.com/CadentTech/porter>`_ and make your changes in a new branch; then `submit a pull request <https://github.com/CadentTech/porter/pulls>`_ for review.

For any contributions, we ask that any new behavior be validated by the test suite (discussed below), which must be passing in its entirety.


Running porter's test suite
---------------------------

To run the test suite for ``porter``, execute the command:

.. code-block:: shell

    make test

Additionally you can install a ``git`` pre-commit hook to run the test suite each time you make a commit with:

.. code-block:: shell

    ./pre-commit-hook install
