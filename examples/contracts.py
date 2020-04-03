import scipy.stats as ss
import pandas as pd
from porter.datascience import BaseModel
from porter.services import ModelApp, PredictionService, BaseService
from porter.schemas import Array, Object, String, Number, Integer
from porter.schemas import openapi


class IdentityModel(BaseModel):
    def predict(self, X):
        return X


identity_model_instance_schema = Object(
    'Inputs to the content recommendation model',
    properties=dict(
        feature1=Number('The first feature in this model is numeric.'),
        feature2=Number('So is the second feature.'),
        feature3=String('The third feature is a string but this description is equally undescriptive.')
    ),
    reference_name='IdentityModelInstance'
)


instance_prediction_service = PredictionService(
    model=IdentityModel(),
    name='my-model',
    api_version='v2',
    namespace='datascience',
    instance_schema=identity_model_instance_schema)

batch_prediction_service = PredictionService(
    model=IdentityModel(),
    name='my-model',
    api_version='v2',
    action='batchPrediction',
    namespace='datascience',
    instance_schema=identity_model_instance_schema,
    batch_prediction=True)


class ProbabilisticModel(BaseModel):
    def predict(self, X):
        dist = ss.norm(X, 1)
        return pd.DataFrame({
            'lower_bound': dist.ppf(0.05),
            'point_estimate': dist.mean(),
            'upper_bound': dist.ppf(0.95),
        })


probabilistic_model_instance_schema = Object(
    'These are objects the probabilistic model will make predictions on.',
    properties={
        'featureA': String('Feature A is a string representing a categorical value.'),
        'featureB': Number('Feature B is another Number.')
    },
    reference_name='ProbaModelInstance'
)


probabilistic_model_prediction_schema = Object(
    'Return a prediction with upper and lower bounds',
    properties={
        'lower_bound': Number('Lower bound on the prediction. '
                              'Actual values should fall below this range just 5% of the time'),
        'point_estimate': Number('The average value we expect actual values to take.'),
        'upper_bound': Number('Upper bound on the prediction. '
                              'Actual values should fall above this range just 95% of the time'),
    },
    reference_name='ProbaModelPrediction'
)

probabilistic_service = PredictionService(
    model=ProbabilisticModel(),
    name='proba-model',
    api_version='v3',
    namespace='datascience',
    instance_schema=probabilistic_model_instance_schema,
    prediction_schema=probabilistic_model_prediction_schema,
    batch_prediction=True
)


class CustomService(BaseService):
    action = 'foo'

    def serve(self):
        pass

    def status(self):
        return 'READY'

custom_service_contracts = [
    openapi.Contract(
        'POST',
        request_schema=openapi.RequestBody(
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
            openapi.ResponseBody(status_code=200, obj=Array(item_type=String())),
            openapi.ResponseBody(status_code=422, obj=Object(properties={'message': String()}))
        ],
        additional_params={'tags': ['custom-service']}
    )
]


custom_service = CustomService(
    name='custom-service',
    api_version='v1',
    api_contracts=custom_service_contracts
)




app = ModelApp(name='Example Model',
               description='An unhelpful description of what this application.',
               expose_docs=True)
app.add_services(instance_prediction_service, batch_prediction_service,
                 probabilistic_service, custom_service)


if __name__ == '__main__':
    app.run()
