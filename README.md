# porter

What is `porter`? `porter` is a framework for exposing machine learning models via REST APIs. Any object with
a `.predict()` method will do which means `porter` plays nicely with models you have already trained
using [sklearn](https://scikit-learn.org/stable/), [keras](https://keras.io/backend/) or [xgboost](https://xgboost.readthedocs.io/en/latest/) to name a few well known machine learning libraries. It also allows
you to easily expose custom models.

Getting started is as easy as

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

Now just send a POST request to the endpoint `/my-model/v1/prediction` to get a prediction. Behind the
scenes (with `porter`s default settings) your POST data will be converted to a `pandas.DataFrame` and
the result of `my_model.predict()` will be returned to the user in a payload like the one below

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

`porter` also takes care of a lot of the boilerplate for you such as error handling. For example, by default,
if the POST data sent to the prediction endpoint can't be parsed the user will receive a response with a 400
status code a payload describing the error.

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

# Installation
`porter` can be installed with `pip` as follows

```shell
pip install -e git+https://github.com/CadentTech/porter#egg=porter
```

Note that without the `-e` flag and `#egg=porter` on the end of the url `pip freeze` will output `porter==<version>`
rather than `-e git+https://...` as typically desired.

If you want to install `porter` from a specific commit or tag, e.g. tag `1.0.0` simply add
`@<commit-or-tag>` immediately before `#egg=porter`.

```shell
pip install -e git+https://github.com/CadentTech/porter@1.0.0#egg=porter
```

For more details on this topic see [here](https://codeinthehole.com/tips/using-pip-and-requirementstxt-to-install-from-the-head-of-a-github-branch/)

## Installing extra dependencies
Porter supports functionality for loading `sklearn` and `keras` models either from disk or `S3`.
Since we don't expect `porter` applications to necessarily need all of these dependencies at once
they can be optionally installed by issuing the command from the root directory of this repository

```shell
pip install -e git+https://github.com/CadentTech/porter#egg=porter[keras-utils,sklearn-utils,s3-utils]
```
Note that you can choose to install only a subset of these additional requirements by removing
the undesired names from the comma separated list in brackets above.

# Usage
The basic workflow for building a model service is as follows

1. Instantiate an instance of `porter.services.ModelApp`. This is simply an abstraction for
  the REST app that will expose your models.
2. Define model classes for each service you want to add to the app. A single service consists of
  a model that satisfies the `porter.datascience.BaseModel` interface. Additionally you can define
  processor classes (objects implementing the `porter.datascience.BaseProcessor` interface) for
  pre/post processing of model input/output respectively. If you have a serialized `sklearn` and/or
  `keras` objects and/or your model is on S3, classes in `porter.datascience` can help load these
  objects.
3. Instantiate classes, such as `porter.services.PredictionService`, with the appropriate arguments for
  each model you would expose through the app.
4. Pass the config service from 3. to the `add_serivce` method of your `ModelApp` instance.
5. Call the `run` method of your `ModelApp` instance. Your model is now live!

See this [example script](./examples/example.py) for an (almost functional) example.

# Running porter apps in production

There are two ways to run porter apps. The first is calling the `ModelApp.run` method. This
is just a wrapper to the underlying `flask` app which is good for development but not for
production. A better way to run porter apps in production is through a production-grade WSGI server
, such as `gunicorn`. To do so simply define an instance of `ModelApp` in your python script and 
then point gunicorn to it.

For example, in your python script `app.py`

```python
model_app = ModelApp(...)
```

Then for production use, either in a shell script or on the command line

```shell
gunicorn app:model_app
```

# API

The complete details of an API exposed by `porter` can be found in the [openapi](https://openapi.tools/) spec
[in this repository](./docs/porter-api.yaml).

Additionally you can open the [static documentation](./docs/html/index.html) generated from this spec
in your web browser. On a Mac

```shell
open docs/html/index.html
```

For those who just want to get a sense of the API a quick overview is below.

## Quick guide

Two healtch check endpoints are exposed by each `porter` app: `/-/alive` and `/-/ready`. These
are useful for deploying a `porter` app in an environment like Kubernets or behind a load balancer.

Additionally there is a prediction endpoint for each model service added to the `ModelApp` instance.
The endpoint is computed from the name and version attributes of the model services:
`/<model name>/<model version>/prediction`. For example

```python
service1 = PredictionService(name='foo', version='v1', ...)
service2 = PredictionService(name='bar', version='v2', ...)
model_app.add_services(service1, service2)
```
will expose two models on the endpoints `/foo/v1/prediction` and `/foo/v2/prediction`. The endpoints
accept POST requests with JSON payloads.

# Logging
API calls (request and response payloads) can be logged by passing `log_api_calls=True` when instantiating
a service. The following data is available for logging

- "request_id": A unique ID for the request.
- "request_data": The request's JSON payload.
- "response_data": The response's JSON payload.
- "service_class": The name of the service class that served the request.
- "event": The type of event being logged, e.g. "request" or "response".

The script [api_logging.py](./examples/api_logging.py) demonstrates how to configure logging.

# Tests
To run the test suite for porter execute the command

```shell
make test
```

Additionally you can install a `git` pre-commit hook to run the test suite each time you make a
commit with

```shell
./pre-commit-hook install
```
