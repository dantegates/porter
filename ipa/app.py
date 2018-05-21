import json

import numpy as np

from ipa import constants as cn
from ipa.datascience import WrappedModel, BaseFeatureEngineer
from ipa.services import ModelApp, ServiceConfig

model_service = ModelApp()


class FeatureEngineer(BaseFeatureEngineer):
    def transform(self, X):
        X['network_id'] = X.network_id.astype(str)
        X['daypart_id'] = X.daypart_id.astype(str)
        X['mvpd_id'] = X.mvpd_id.astype(str)
        X['week_number'] = X.week_number.astype(str)
        X['spot_length_id'] = X.spot_length_id.astype(str)
        X['rate_card_type_id'] = X.rate_card_type_id.astype(str)
        X['log_delivery_goal'] = np.log10(X.log_delivery_goal)
        return X


input_schema = json.load(cn.INPUT_SCHEMA_PATH)
feature_engineer = FeatureEngineer()
model = WrappedModel.from_file(
    path=cn.MODEL_PATH,
    id=cn.MODEL_ID,
    name=cn.MODEL_NAME)
service_config = ServiceConfig(
    model=model,
    feature_engineer=feature_engineer,
    input_schema=input_schema,
    check_input=True,
    allow_nulls=False)
model_service.add_service(service_config)


if __name__ == '__main__':
    # you can run this with `gunicorn app:model_service.app`
    # localhost:8000/preflight-index-model/prediction <- POST requests
    model_service.app.run(port=8000)
