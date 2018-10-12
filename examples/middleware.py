from porter.datascience import BaseModel
from porter.services import ModelApp, PredictionServiceConfig, MiddlewareServiceConfig


app = ModelApp()


class Model(BaseModel):
    def predict(self, X):
        return (X['foo'] % 3) * X['bar']

prediction_config = PredictionServiceConfig(model=Model(),
                                            name='my-model',
                                            version='1',
                                            batch_prediction=False)
middleware_config = MiddlewareServiceConfig(name='my-model',
                                            version='1',
                                            max_workers=None,  # use default
                                            model_endpoint=f'http://localhost:5000{prediction_config.endpoint}')

app.add_services(prediction_config, middleware_config)


if __name__ == '__main__':
    app.run()
