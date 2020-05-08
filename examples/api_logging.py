"""Test with

curl localhost:5000/-/alive | python -m json.tool
curl localhost:5000/my-model/v1/prediction -d @examples/api_logging.json | python -m json.tool
"""

import logging

from porter.datascience import BaseModel
from porter.services import ModelApp, PredictionService
from porter.utils import JSONLogFormatter


class Model(BaseModel):
    def predict(self, X):
        return (X['foo'] % 3) * X['bar']


prediction_svc = PredictionService(
    model=Model(),
    name='my-model',
    api_version='v1',
    batch_prediction=True,
    log_api_calls=True)


app = ModelApp([prediction_svc])


if __name__ == '__main__':
    import logging
    stream_handler = logging.StreamHandler()
    formatter = JSONLogFormatter(
        'asctime', 'levelname', 'module', 'name', 'message',
        'request_id', 'request_data', 'response_data', 'service_class',
        'event')
    stream_handler.setFormatter(formatter)
    logger = logging.getLogger('porter')
    logger.setLevel('INFO')
    logger.addHandler(stream_handler)
    app.run()
