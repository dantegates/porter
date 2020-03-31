from porter.services import ModelApp, PredictionService
from porter.schemas import Array, Object, String, Number, Contract


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
PostContract = Contract('POST', request_schema=request_schema,
                        response_schemas=None, validate_request_data=True)


class IdentityModel:
    def predict(self, X):
        return X


prediction_service = PredictionService(
    model=IdentityModel(),
    name='my-model',
    api_version='v2',
    namespace='datascience',
    api_contracts=[PostContract])

app = ModelApp(expose_docs=True)
app.add_service(prediction_service)


if __name__ == '__main__':
    app.run()
