"""Test with

curl localhost:5000/-/alive | python -m json.tool

curl -d @examples/middleware.json -POST localhost:5000/my-model/batchPrediction

"""


from porter.datascience import BaseModel
from porter.services import ModelApp, PredictionService, MiddlewareService


app = ModelApp()


class Model(BaseModel):
    def predict(self, X):
        return (X['foo'] % 3) * X['bar']


prediction_svc = PredictionService(
    model=Model(),
    name='my-model',
    api_version='1',
    batch_prediction=False)
middleware_svc = MiddlewareService(
    name='my-model',
    api_version='1',
    max_workers=None,  # use default
    model_endpoint=f'http://localhost:5000{prediction_svc.endpoint}')
middleware_svc_never_ready = MiddlewareService(
    name='another-model',
    api_version='1',
    max_workers=None,  # use default
    model_endpoint=f'http://localhost:8000/does-not-exist')
app.add_services(prediction_svc, middleware_svc, middleware_svc_never_ready)


if __name__ == '__main__':
    app.run()
