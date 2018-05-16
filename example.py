import datascience as ds
from services import ModelService


model_service = ModelService()

model = ds.BaseModel.from_file('path/to/model.pkl')
feature_engineer = ds.BaseFeatureEngineer.from_file('path/to/feature_engineer.pkl')
another_model = ds.BaseModel.from_file('path/to/another_model.h5')
model_service.route_model(model, 'super-model', feature_engineer=feature_engineer)
model_service.route_model(another_model, 'another-super-model')


if __name__ == '__main__':
    # localhost:8888/super-model/prediction <- POST requests
    # localhost:8888/another-super-model/prediction <- POST requests
    model_service.run(port=8888)
