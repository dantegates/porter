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

ratings_feature_schema = Object(
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
`ratings_feature_schema` contains a `validate()` method which can be used to
validate Python objects against the schema. Keep in mind that you never need
to call this method explicitly yourself, `porter` will automatically validate
`POST` data for you.
"""

# no error raised
ratings_feature_schema.validate({
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
    ratings_feature_schema.validate({
        'user_id': 1,
        'title_id': 1,
        'genre': 'not an acceptable value',
        'average_rating': 8.9
    })
except Exception as err:
    print(err)

try:
    ratings_feature_schema.validate({
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
according to `ratings_feature_schema`. Validations can be disabled by setting
`validate_request_data=False`.
"""


instance_prediction_service = PredictionService(
    model=RatingsModel(),
    name='user-ratings',
    api_version='v2',
    namespace='datascience',
    feature_schema=ratings_feature_schema,
    validate_request_data=True)


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
    feature_schema=ratings_feature_schema,
    batch_prediction=True,
    validate_request_data=True)


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
    reference_name='ProbaModelPrediction')


probabilistic_service = PredictionService(
    model=ProbabilisticRatingsModel(),
    name='proba-model',
    api_version='v3',
    namespace='datascience',
    feature_schema=ratings_feature_schema,
    prediction_schema=proba_ratings_prediction_schema,
    batch_prediction=True)


"""
While ``feature_schema`` and ``prediction_schema`` parameters allow users to
specify the input and output schemas for a successful POST request that
returns predictions, some use cases may require the ability to specify
response schemas for non-200 status codes.

For example, perhaps now your recommendations are generated from a big matrix
factorization job executed in some other environment, e.g. Spark. Because the
predictions are now happening in another, perhaps long-running, compute
environment a suitable contract is to return a response with a 202 status code
and an ID that can be used to retrieve the predictions later after the job
finishes.

Fortunately, we can still use ``porter`` as an interface to the Spark job, but
it will take a bit more work. We can extend the :class:`porter.services.PredictionService`
class to validate and the request schemas as before, while customizing the
documentation to include a 202 response for POST requests as follows.
"""


from porter.responses import Response


class SparkInterfaceModel(BaseModel):
    def predict(self, X):
        job_id = self._generate_job_id()
        self._submit_job(X, job_id)
        return job_id

    def _generate_job_id(self):
        return 1

    def _submit_job(self, X, job_id):
        pass


class SparkInterfaceService(PredictionService):
    def serve(self):
        X = self.get_post_data()  # validations are done here if validate_request_data=True
        job_id = self.model.predict(X)
        return Response({'job_id': job_id}, status_code=202)


spark_interface_response_schema = Object(
    properties={'job_id': Integer('ID used to retrieve results from batch job.')},
    reference_name='BatchRatingsOutput'
)

spark_interface_service = SparkInterfaceService(
    model=SparkInterfaceModel(),
    name='batch-ratings-model',
    api_version='v1',
    namespace='datascience',
    feature_schema=ratings_feature_schema,
    validate_request_data=True,
    batch_prediction=True)

# note that when we specify schemas directly, ``porter`` leaves them untouched,
# unlike when they are specified via the "convenience" properties ``feature_schema``
# and ``prediction_schema``.
spark_interface_service.add_response_schema('POST', 202, spark_interface_response_schema)


"""
What if we need even more control to define a completely customized API?

``porter`` supports documentation and validation of arbitrary schemas, however
you will have to define a custom service class and using quite a few more
objects from `porter.schemas` if the inputs and outputs can't be represented
in a "tabular" way.

Here's an example.
"""


class CustomService(BaseService):
    action = 'foo'
    route_kwargs = {'methods': ['GET', 'POST']}

    def serve(self):
        data = self.get_post_data()
        return {'results': ['foo', 'bar']}

    def status(self):
        return 'READY'


"""
Defining a very nested, customized data structure.
"""

from porter.schemas import request_id, model_context


custom_service_input = Object(
    properties={
        'string_with_enum_prop': String(additional_params={'enum': ['a', 'b', 'abc']}),
        'an_array': Array(item_type=Number()),
        'another_property': Object(properties={'a': String(), 'b': Integer()}),
        'yet_another_property': Array(item_type=Object(additional_properties_type=String()))
    },
    reference_name='CustomServiceInputs'
)

custom_service_output_success = Object(
    properties={
        'request_id': request_id,
        'model_context': model_context,
        'results': Array(item_type=String())
    }
)

custom_service = CustomService(name='custom-service', api_version='v1', validate_request_data=True)
custom_service.add_request_schema('POST', custom_service_input)
custom_service.add_response_schema('POST', 200, custom_service_output_success)


"""
Notice how the response objects define "request_id" and "model_context'
properties to agree with ``porter``s default response objects.

If you find that you need to create highly custom schemas like this, be sure
to first understand these defaults.

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
                       probabilistic_service, spark_interface_service,
                       custom_service)



"""
These are just some convenience functions to test the example.
"""


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

