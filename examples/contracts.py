from porter.services import ModelApp, PredictionService
from porter.schemas import Array, Object, String, Number


request_schema = Array(
    'Inputs to the content recommendation model',
    item_type=Object(
        'Instance',
        properties=dict(
            feature1=Number(),
            feature2=Number(),
            feature3=String()
        )
    )
)


class IdentityModel:
    def predict(self, X):
        return X


prediction_service = PredictionService(
    model=IdentityModel(),
    name='my-model',
    api_version='v2',
    namespace='datascience',
    request_schema=request_schema)

app = ModelApp(name='Example Model',
               description='An unhelpful description of what this model does',
               expose_docs=True)
app.add_service(prediction_service)


if __name__ == '__main__':
    app.run()
