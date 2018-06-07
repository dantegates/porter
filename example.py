
"""
This code demonstrates how to expose a pickled sklearn model as a REST
API via porter.

The model predictions can be obtained by sending POST requests with a payload
such as

    [
        {"id": 101, "feature1": "foo", "feature2": "bar", "feature3": 1.0}
    ]

to the endpoint

    <host>:<port>/supa-dupa-model/prediction/

The corresponding output has the format

    [
        "model_id": 'supa-dupa-model-v0',
        "predicitons": [
            {"id": 101, "prediction": 1001.01}
        ]
    ]
"""

from porter.datascience import WrappedModel, WrappedTransformer, BaseProcessor
from porter.services import ModelApp, ServiceConfig


# first we instantiate the model app.
# The model app is simply a wrapper around the `flask.Flask` object.
#
# Services are added to the app with `model_app.add_service` below.
model_app = ModelApp()

# define the expected input schema so the model can validate the POST
# request input
input_schema = [
    'feature1',
    'feature2',
    'column3',
]

# Define a preprocessor, model and postprocessor for transforming the
# POST request data, predicting and transforming the model's predictions.
# Both processor instances are optional.
#
# For convenience we can load pickled `sklearn` objects as the preprocessor
# and model.
class Postprocessor(BaseProcessor):
    def process(self, X):
        # model predicts the log of value we really care about
        return 10**X
postprocessor = Postprocessor()
preprocessor = WrappedTransformer.from_file(path='/path/to/feature_engineer.pkl')
model = WrappedModel.from_file(path='/path/to/model.pkl')

# the service config contains everything needed for `model_app` to add a route
# for predictions when `model_app.add_service` is called.
service_config = ServiceConfig(
    model=model,                        # The value of model.predict() is
                                        # returned to the client.
                                        # Required.
                                        #
    endpoint='supa-dupa-model',         # Name of the model. This determines
                                        # the route. E.g. send POST requests
                                        # for this model to
                                        #   host:port/supa-dupa-model/prediction/
                                        # Required.
                                        #
    model_id='supa-dupa-model-v0',      # Unique identifier for the model. Returned
                                        # to client in the prediction response.
                                        # Required.
                                        #
    preprocessor=preprocessor,          # preprocessor.transform() is
                                        # called on the POST request data
                                        # before predicting. Optional.
                                        #
    postprocessor=postprocessor,        # postprocessor.transform() is
                                        # called on the model's predictions before
                                        # returning to user. Optional.
                                        #
    input_schema=input_schema,          # The input schema is used to validate
                                        # the payload of the POST request.
                                        # Optional.
                                        #
    allow_nulls=False                   # Wether nulls are allowed in the POST
                                        # request data. Optional and meaningless
                                        # when validate_input=False.
)

# The model can now be added as a service in the app.
model_app.add_service(service_config)


if __name__ == '__main__':
    # you can run this with `gunicorn app:model_app.app`, or
    # simply execute this script with Python and send POST requests
    # to localhost:8000/supa-dupa-model/prediction/
    model_app.run(port=8000)
