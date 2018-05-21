from porter.datascience import WrappedModel, BaseFeatureEngineer
from porter.services import ModelApp, ServiceConfig


# first we instantiate the model app.
# The model app is simply a wrapper around the `flask.Flask` object.
#
# Services are added to the app with `model_app.add_service` below.
model_app = ModelApp()


input_schema = {
    'column1': 'str',
    'column2': 'str',
    'column3': float
}
feature_engineer = WrappedFeatureEngineer.from_file(path='/path/to/feature_engineer.pkl')
model = WrappedModel.from_file(path='/path/to/model.pkl')

# the service config contains everything needed for `model_app` to add a route
# for predictions when `model_app.add_service` is called.
service_config = ServiceConfig(
    model=model,                        # The value of model.predict() is
                                        # returned to the client.
                                        # Required.
                                        #
    model_name='supa-dupa-model',       # Name of the model. This determines
                                        # the route. E.g. send POST requests
                                        # for this model to
                                        #   host:port/supa-dupa-model/prediction/
                                        # Required.
                                        #
    model_id='supa-dupa-model-v0',      # Unique identifier for the model. Returned
                                        # to client in the prediction response.
                                        # Required.
                                        #
    feature_engineer=feature_engineer,  # feature_engineer.transform() is
                                        # called on the POST request data
                                        # before predicting. Optional.
                                        #
    input_schema=input_schema,          # The input schema is used to validate
                                        # the payload of the POST request.
                                        # Optional.
                                        #
    validate_input=True,                # Payload input validated only when this
                                        # is True. Optional and False by default.
                                        #
    allow_nulls=False                   # Wether nulls are allowed in the POST
                                        # request data. Optional and meaningless
                                        # when validate_input=False.
)

# The model can now be added as a service in the app.
model_app.add_service(service_config)


if __name__ == '__main__':
    # you can run this with `gunicorn app:model_app.app`
    # localhost:8000/preflight-index-model/prediction <- POST requests
    model_app.app.run(port=8000)
