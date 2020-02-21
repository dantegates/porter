# porter

<p align="center">
[![Documentation Status](https://readthedocs.org/projects/porter/badge/?version=latest)](https://porter.readthedocs.io/en/latest/?badge=latest)
</p>

What is `porter`? `porter` is a framework for exposing machine learning models
via REST APIs. Any object with a `.predict()` method will do which means
`porter` plays nicely with models you have already trained using
[sklearn](https://scikit-learn.org/stable/), [keras](https://keras.io/backend/)
or [xgboost](https://xgboost.readthedocs.io/en/latest/) to name a few well
known machine learning libraries. It also includes the ability to load `.pkl`
and `.h5` files so you don't have to write this code every time you deploy a
new model and allows you to easily expose custom models.  Getting started is as
easy as

```python
from porter.datascience import WrappedModel
from porter.services import ModelApp, PredictionService

my_model = WrappedModel.from_file('my-model.pkl')
prediction_service = PilotPredictionService(
    model=my_model,
    name='my-model',
    api_version='v1')

app = ModelApp()
app.add_service(prediction_service)
app.run()
```

Now just send a POST request to the endpoint `/my-model/v1/prediction` to get a
prediction. Behind the scenes (with `porter`s default settings) your POST data
will be converted to a `pandas.DataFrame` and the result of
`my_model.predict()` will be returned to the user in a payload like the one
below

```javascript
{
    "model_context": {
        "api_version": "v1",
        "model_meta": {},
        "model_name": "my-model"
    },
    "predictions": [
        {
            "id": 1,
            "prediction": 0
        }
    ],
    "request_id": "0f86644edee546ee9c495a9a71b0746c"
}
```

`porter` also takes care of a lot of the boilerplate for you such as error
handling. For example, by default, if the POST data sent to the prediction
endpoint can't be parsed the user will receive a response with a 400 status
code a payload describing the error.

```javascript
{
    "error": {
        "messages": [
            "The browser (or proxy) sent a request that this server could not understand."
        ],
        "name": "BadRequest"
    },
    "request_id": "852ca09d578b447aa3d41d70b8cc4431"
}
```

For more information, see the [documentation](https://porter.readthedocs.org).
