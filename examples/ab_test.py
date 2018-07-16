from porter.datascience import BaseModel
from porter.services import ABTestConfig, ModelApp, PredictionServiceConfig

# first we instantiate the model app.
# The model app is simply a wrapper around the `flask.Flask` object.
#
# Services are added to the app with `model_app.add_service` below.
model_app = ModelApp()

class Model(BaseModel):
    def __init__(self, *args, val, **kwargs):
        self.val = val
        super().__init__(*args, **kwargs)

    def predict(self, X):
        return self.val


service_configs = []
for val in ['A', 'B', 'C']:
    config = PredictionServiceConfig(model=Model(val=val),
                                     id=f'model_{val.lower()}',
                                     input_features=[])
    service_configs.append(config)

ab_test_config = ABTestConfig(service_configs, splits=[0.1, 0.2, 0.7],
                              endpoint_basename='supa-dupa-model',
                              id='supa-dupa-model-ab-test')

model_app.add_service(ab_test_config)


if __name__ == '__main__':
    # you can run this with `gunicorn app:model_app.app`, or
    # simply execute this script with Python and send POST requests
    # to localhost:8000/supa-dupa-model/prediction/
    model_app.run(port=8000)
