.. _openapi_schemas:

OpenAPI Schemas
===============

``porter`` provides support for automatically validating and documenting payload schemas.  Validation and documentation are built on a shared schema specification framework, lending confidence that the documentation accurately reflects models being served.

.. note::
    These features require the optional ``schema-validation`` dependency to be enabled.


Schema Definition
-----------------

``porter`` supports the `OpenAPI Specification <https://swagger.io/docs/specification/about/>`_.  Schema definition is facilitated by :mod:`porter.schemas`.  Valid data types include:

.. code-block:: python

    from porter.schemas import Boolean, Integer, Number, String, Array, Object

For each type, the first argument is an optional (but recommended) description.
Objects take a ``properties`` argument of type ``dict`` specifying the names and types of each nested element.  All types take an optional ``additional_params`` argument, also of type ``dict``, giving access to all other keys in the OpenAPI specification.

For example, the input expected by a :class:`porter.services.PredictionService` serving a rating prediction model might look like so:

.. code-block:: python

    feature_schema = Object(
        'Inputs to the ratings model',
        properties=dict(
            user_id=Integer('The user ID.'),
            title_id=Integer('The title ID.'),
            is_tv=Boolean('Whether the content is a TV show.'),
            genre=String('The genre.',
                         additional_params={'enum': ['comedy', 'action', 'drama']}),
            average_rating=Number('The title\'s average rating.',
                                  additional_params={'minimum': 0, 'maximum': 10}),
        ),
        reference_name='RatingsModelFeatures'
    )

This schema is equivalent to the following yaml markup:

.. code-block:: yaml

    - RatingsModelFeatures:
        type: object
        description: Inputs to the ratings model
        properties:
          user_id:
            type: integer
            description: The user ID.
          title_id:
            type: integer
            description: The title ID.
          is_tv:
            type: boolean
            description: Whether the content is a TV show.
          genre:
            type: string
            description: The genre.
            enum: [comedy, action, drama]
          average_rating:
            type: number
            description: The title's average rating.
            minimum: 0
            maximum: 10
        required: [average_rating, genre, is_tv, title_id, user_id]


``PredictionService`` adds a required integer field, ``id``, to the schema.  Also, by default, ``PredictionService`` performs batch prediction over multiple objects, and thus the above would become the item type for an Array.  These modifications are equivalent to:

.. code-block:: python

    instance_schema = Object(properties={'id': Integer(), **feature_schema.properties})
    batch_schema = Array(item_type=instance_schema)

Notice that here ``item_type`` is another API object type, in this case ``Object``.  Both :attr:`Array.item_type` and :attr:`Object.properties` are composable in this way, and will be implemented using OpenAPI ``$ref`` if ``reference_name`` is given.



Schema Validation
-----------------

We can add input validation against the above schema to the PredictionService in :ref:`getting_started` like so:

.. code-block:: python

    prediction_service = PredictionService(
        model=my_model,
        name='my-model',
        api_version='v1',
        feature_schema=feature_schema,
        validate_request_data=True)

Now, for valid input such as

.. code-block:: json

    [
        {
            "id": 1,
            "user_id": 122333,
            "title_id": 444455555,
            "is_tv": true,
            "genre": "comedy",
            "average_rating": 6.7
        },
        {
            "id": 2,
            "user_id": 122333,
            "title_id": 788999,
            "is_tv": false,
            "genre": "drama",
            "average_rating": 4.3
        }
    ]

we receive predictions as expected, but input such as

.. code-block:: json

    [
        {
            "id": 1,
            "user_id": 122333,
            "title_id": 444455555,
            "genre": "not-a-real-genre",
            "average_rating": 6.7
        },
    ]

will result in a 422 error (Unprocessable Entity).  Error handling is discussed further in :ref:`this section <error_handling>`.


Schema Documentation
--------------------

To expose `Swagger <https://swagger.io/>`_ documentation automatically, simply ``expose_docs=True`` to the :class:`porter.services.ModelApp` constructor.  We'll also set the ``name`` and ``description`` attributes, which will appear in the documentation.

.. code-block:: python

    app = ModelApp(
        [prediction_service],
        name='Example Model',
        description='Minimal example of a model with input validation and documentation.',
        expose_docs=True)


If this app is run in testing mode, docs are now available at ``http://localhost:5000/docs/``.  The top of the page shows the name and description of the app, followed by information about the exposed endpoints:

.. image:: _static/swagger_main.png
    :width: 80%
    :alt: Auto-generated API documentation -- main interface.
    :align: center

At the bottom of the page, we find a list of schemas which can be unfolded and inspected:

.. image:: _static/swagger_schemas.png
    :width: 80%
    :alt: Auto-generated API documentation -- schema list.
    :align: center

The endpoint documentation can be unfolded, and you can select "Try it out" to test it:

.. image:: _static/swagger_tryitout.png
    :width: 80%
    :alt: Auto-generated API documentation -- "try it out" feature.
    :align: center
