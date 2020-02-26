# porter

[![Documentation Status](https://readthedocs.org/projects/porter/badge/?version=latest)](https://porter.readthedocs.io/en/latest/?badge=latest)
What is `porter`? `porter` is a framework for exposing machine learning models
behind a REST API. Any object with a `.predict()` method will do which means
`porter` plays nicely with models you have already trained using
[sklearn](https://scikit-learn.org/stable/), [keras](https://keras.io/backend/)
or [xgboost](https://xgboost.readthedocs.io/en/latest/) to name a few well
known machine learning libraries. In addition `porter` also seeks to reduce the amount of
boiler plate you need to write, e.g. it includes the ability to load `.pkl`
and `.h5` files so you don't have to write this code each time you deploy a
new model. Getting started is as easy as

```python
# myapp.py
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

To get predictions, simply run the script above and send a POST request to
the endpoint `localhost:5000/my-model/v1/prediction`. Behind the scenes
`porter` will convert your POST data to a `pandas.DataFrame`, pass the data
off to `my_model.predict()` and return the results.

```shell
python myapp.py &
curl -POST -d '[{"feature1": 1, "feature2": 2.2}]' localhost:5000/my-model/v1/prediction
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

`porter` apps are [WSGI](https://wsgi.readthedocs.io/en/latest/learn.html) apps
which means they can be responsibly deployed into production environments with
software like [gunicorn](https://gunicorn.org/).


# Installation

`porter` can be installed with pip as follows

```shell
pip install -e git+https://github.com/CadentTech/porter#egg=porter
```

Note that without the `-e` flag and `#egg=porter` on the end of the url
`pip freeze` will output `porter==<version>` rather than
`-e git+https://...` as typically desired.

If you want to install porter from a specific commit or tag, e.g. tag `1.0.0` simply add
`@<commit-or-tag>` immediately before `#egg=porter`. E.g.

```shell
pip install -e git+https://github.com/CadentTech/porter@1.0.0#egg=porter
```

For more details on this, see [this
page](https://porter.readthedocs.io/en/latest/installation.html).

# Documentation
For more information, see the [documentation](https://porter.readthedocs.org).

Copyright (c) 2020 Cadent Data Science


