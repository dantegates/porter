"""Self-contained working example demonstrating typical usage."""

import pickle
from porter.datascience import WrappedModel
from porter.services import ModelApp, PredictionService
from porter.schemas import Boolean, Integer, Number, String, Object


# for this example to run, we need to make up a trivial my-model.pkl
class MyModel:
    def predict(self, X):
        return X['average_rating']
with open('my-model.pkl', 'wb') as f:
    pickle.dump(MyModel(), f, -1)

# now we load the model as a WrappedModel
my_model = WrappedModel.from_file('my-model.pkl')

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
