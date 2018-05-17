class PredictionResponse(dict):
    def __init__(self, model_id, id_keys, predictions):
        response = self.make_response(model_id, id_keys, predictions)
        super(PredictionResponse, self).__init__(response)

    @staticmethod
    def make_response(model_id, id_keys, predictions):
        return {
            'model_id': model_id,
            'predictions': [{id: p} for id, p in zip(id_keys, predictions)]
        }
