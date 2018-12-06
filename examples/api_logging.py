"""Test with

curl localhost:5000/-/alive | python -m json.tool

curl -d @examples/middleware.json -POST localhost:5000/my-model/batchPrediction

"""

import logging

from porter.datascience import BaseModel
from porter.services import ModelApp, PredictionService, MiddlewareService
from porter.utils import JSONFormatter


app = ModelApp()


class Model(BaseModel):
    def predict(self, X):
        return (X['foo'] % 3) * X['bar']


prediction_svc = PredictionService(
    model=Model(),
    name='my-model',
    version='1',
    batch_prediction=False,
    log_api_calls=True)
middleware_svc = MiddlewareService(
    name='my-model',
    version='1',
    max_workers=None,  # use default
    model_endpoint=f'http://localhost:5000{prediction_svc.endpoint}')
app.add_services(prediction_svc, middleware_svc)


if __name__ == '__main__':
    import logging
    stream_handler = logging.StreamHandler()
    formatter = JSONFormatter(
        'asctime', 'levelname', 'module', 'name', 'message',
        'request_id', 'data', 'service_class', 'event')
    stream_handler.setFormatter(formatter)
    logger = logging.getLogger('porter')
    logger.setLevel('INFO')
    logger.addHandler(stream_handler)
    app.run()
