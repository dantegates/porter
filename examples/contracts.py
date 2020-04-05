import os
import sys

import numpy as np
import pandas as pd
import scipy.stats as ss
from porter import schemas
from porter.datascience import BaseModel
from porter.schemas import Array, Integer, Number, Object, String
from porter.services import BaseService, ModelApp, PredictionService


"""
Suppose we have a model that predicts the rating a given user would assign
to a particular title.

First, we'll need a model to route requests to. The following model doesn't
do anything other than give us an example we can run, but we can use our
imagination.
"""


class RatingsModel(BaseModel):
    def predict(self, X):
        return np.arange(len(X))


"""
In order to use `porter` to automatically document our API and validate
requests we'll need to instantiate an object that represents a single
instance that the model will predict on.

Suppose our ratings model was trained on the features "user_id", "title_id",
"genre" and "average_rating". The JSON passed to the endpoint might look
something like

{
    "user_id": 122333,
    "title_id": 444455555,
    "genre": "comedy",
    "average_rating": 6.7
}

and the corresponding Python object used by `porter` might look like
"""

ratings_feataure_schema = Object(
    'Inputs to the content recommendation model',
    properties=dict(
        user_id=Integer('The user ID.'),
        title_id=Integer('The title ID.'),
        genre=String('The genre.',
                     additional_params={'enum': ['comedy', 'action', 'drama']}),
        average_rating=Number('The title\'s average rating.',
                              additional_params={'minimum': 0, 'maximum': 10})
    ),
    reference_name='RatingsModelInstance'
)


"""
`ratings_feataure_schema` contains a `validate()` method which can be used to
validate Python objects against the schema. Keep in mind that you never need
to call this method explicitly yourself, `porter` will automatically validate
`POST` data for you.
"""

# no error raised
ratings_feataure_schema.validate({
    'user_id': 1,
    'title_id': 1,
    'genre': 'drama',
    'average_rating': 8.9
})

"""
Notice the usage of `additional_params` which gives us access to the full
range of swagger properties and makes the data above valid and the objects
below invalid.
"""

try:
    ratings_feataure_schema.validate({
        'user_id': 1,
        'title_id': 1,
        'genre': 'not an acceptable value',
        'average_rating': 8.9
    })
except Exception as err:
    print(err)

try:
    ratings_feataure_schema.validate({
        'user_id': 1,
        'title_id': 1,
        'genre': 'drama',
        'average_rating': -1  # outside of range
    })
except Exception as err:
    print(err)



"""
Now we can instantiate a PredictionService for our model and simply pass it
the schema. By default requests sent to this endpoint will be validated
according to `ratings_feataure_schema`. Validations can be disabled by setting
`validate_request_data=False`.
"""


instance_prediction_service = PredictionService(
    model=RatingsModel(),
    name='user-ratings',
    api_version='v2',
    namespace='datascience',
    feature_schema=ratings_feataure_schema)


"""
Because batch prediction is disabled in `porter` APIs by default the following
is a valid payload to `/datascience/user-ratings/v2/prediction`

{
    "id": 1,
    "user_id": 122333,
    "title_id": 444455555,
    "genre": "comedy",
    "average_rating": 6.7
}

Note the "id" property which is required by `porter`. Because this property
is always required, we didn't need to include it in the spec we defined as
`porter` will add it for us.

If we want to support for batch prediction, we can reuse the schema above and
simply specify `batch_prediction=True`. Now we can send requests like


[
    {
        "id": 1,
        "user_id": 122333,
        "title_id": 444455555,
        "genre": "comedy",
        "average_rating": 6.7
    },
    {
        "id": 2,
        "user_id": 122333,
        "title_id": 788999,
        "genre": "drama",
        "average_rating": 4.3
    }
]

to `/datascience/user-ratings/v2/batchPrediction`.
"""


batch_prediction_service = PredictionService(
    model=RatingsModel(),
    name='user-ratings',
    api_version='v2',
    action='batchPrediction',
    namespace='datascience',
    feature_schema=ratings_feataure_schema,
    batch_prediction=True)


"""
Let's consider a more advance example.

Suppose we have a probabilistic version of the ratings model that returns
a dictionary for each prediction instead of a single scalar value.

Here's another model definition that doesn't do anything but give us a working
example.
"""


class ProbabilisticRatingsModel(BaseModel):
    def predict(self, X):
        dist = ss.norm(ss.norm(0, 1).rvs(len(X)), 1)
        return pd.DataFrame({
            'lower_bound': dist.ppf(0.05),
            'expected_value': dist.mean(),
            'upper_bound': dist.ppf(0.95),
        }).to_dict(orient='records')


"""
As when we defined the input schema, here we define only a single prediction
instance and `porter` will take care of adding the "id" property and
determining whether the output is an array or single object.
"""


proba_ratings_prediction_schema = Object(
    'Return a prediction with upper and lower bounds',
    properties={
        'lower_bound': Number('Lower bound on the prediction. '
                              'Actual values should fall below this range just 5% of the time'),
        'expected_value': Number('The average value we expect actual values to take.'),
        'upper_bound': Number('Upper bound on the prediction. '
                              'Actual values should fall above this range just 95% of the time'),
    },
    reference_name='ProbaModelPrediction'
)


probabilistic_service = PredictionService(
    model=ProbabilisticRatingsModel(),
    name='proba-model',
    api_version='v3',
    namespace='datascience',
    feature_schema=ratings_feataure_schema,
    prediction_schema=proba_ratings_prediction_schema,
    batch_prediction=True
)


"""
What if we need to define a completely custom schema?

`porter` supports documentation and validation of arbitrary schemas, however
you will have to define a custom service class and using quite a few more
objects from `porter.schemas` if the inputs and outputs can't be represented
in a "tabular" way.

Here's an example.
"""


class CustomService(BaseService):
    action = 'foo'

    def serve(self):
        pass

    def status(self):
        return 'READY'


"""
Defining a very nested, customized data structure.
"""


custom_service_contracts = [
    schemas.Contract(
        'POST',
        request_schema=schemas.RequestBody(
            Object(
                properties={
                    'string_with_enum_prop': String(additional_params={'enum': ['a', 'b', 'abc']}),
                    'an_arry': Array(item_type=Number()),
                    'another_property': Object(properties={'a': String(), 'b': Integer()}),
                    'yet_another_property': Array(item_type=Object(additional_properties_type=String()))
                },
                reference_name='CustomServiceInputs'
            )
        ),
        response_schemas=[
            schemas.ResponseBody(status_code=200, obj=Array(item_type=String())),
            schemas.ResponseBody(status_code=422, obj=Object(properties={'message': String()}))
        ],
        additional_params={'tags': ['custom-service']}
    )
]


custom_service = CustomService(
    name='custom-service',
    api_version='v1',
    api_contracts=custom_service_contracts
)


"""
The last thing we need to do here is instantiate the model app and let it run.
Be sure to specify `expose_docs=True` or the documentation won't be included.
Note that errors explicitly raised by `porter` will be added to the documentation
as will the health check endpoints.

By default you can find the OpenAPI documentation at the endpoint `/docs/` but
this too can be customized.
"""


model_app = ModelApp(name='Example Model',
                     description='An unhelpful description of what this application.',
                     expose_docs=True)
model_app.add_services(instance_prediction_service, batch_prediction_service,
                       probabilistic_service, custom_service)


class Shhh:
    """Silence flask logging."""

    def __init__(self):
        self.devnull = open(os.devnull, 'w')
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def __enter__(self):
        sys.stdout = self.devnull
        sys.stderr = self.devnull

    def __exit__(self, *exc):
        sys.stdout = self.stdout
        sys.stderr = self.stderr


if __name__ == '__main__':
    print('http://localhost:5000/')
    with Shhh():
        model_app.run()

