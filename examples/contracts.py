import scipy.stats as ss
import pandas as pd
from porter.datascience import BaseModel
from porter.services import ModelApp, PredictionService
from porter.schemas import Array, Object, String, Number


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

app = ModelApp(name='Example Model',
               description='An unhelpful description of what this application.',
               expose_docs=True)
app.add_services(instance_prediction_service, batch_prediction_service, probabilistic_service)


if __name__ == '__main__':
    app.run()
