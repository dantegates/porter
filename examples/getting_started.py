"""Self-contained working example demonstrating typical usage."""

import joblib
from porter.datascience import WrappedModel
from porter.services import ModelApp, PredictionService
from porter.schemas import Boolean, Integer, Number, String, Object


# for this example to run, we need to make up a trivial class
# with a .predict() method
class MyModel:
    def predict(self, X):
        return X['average_rating']

# in a real use case, the trained model would already be saved:
#joblib.dump(MyModel(), 'my-model.pkl')

# and we could load it with WrappedModel.from_file():
#my_model = WrappedModel.from_file('my-model.pkl')

# but to make this script testable, we pass a MyModel instance directly:
my_model = WrappedModel(MyModel())

# define the feature schema
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

# build the prediction service
prediction_service = PredictionService(
    model=my_model,
    name='my-model',
    api_version='v1',
    feature_schema=feature_schema,
    validate_request_data=True)

app = ModelApp(
    [prediction_service],
    name='Example Model',
    description='Minimal example of a model with input validation and documentation.',
    expose_docs=True)

if __name__ == '__main__':
    app.run()
