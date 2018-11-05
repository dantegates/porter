# porter
porter is a framework for exposing machine learning models via REST APIs.

# Installation
`porter` can be installed with `pip` as follows

```shell
pip install -e git+https://github.com/CadentTech/porter#egg=porter
```

Note that without the `-e` flag and `#egg=porter` on the end of the url `pip freeze` will output `porter==<version>`
rather than `-e git+https://...` as typically
desired.

If you want to install `porter` from a specific commit or tag, e.g. tag `1.0.0` simply and 
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
production. A better way to run porter apps in production is through a WSGI server, such as
`gunicorn`. To do so simply define an instance of `ModelApp` in your python script and then
point gunicorn to it.

For example, in your python script `app.py`

```python
model_app = ModelApp(...)
```

Then for production use, either in a shell script or on the command line

```shell
gunicorn app:model_app
```

# API
A `porter` defines the following endpoints.

**/-/alive** (Methods=`[GET]`):
  An endpoint used to determine if the app is alive (i.e. running). This endpoint returns the
  same JSON payload returned by **/-/ready**. Returns 200.
  
**/-/ready** (Methods=`[GET]`):
  Returns a JSON object representing the app's state as follows. The object has a single key
  `"services"`. Services is itself a JSON object with a key for every service added to the app.
  These service objects contain keys for their respective endpoint and status. Returns 200 if
  all services are ready and 503 otherwise.

  The JSON below is the response you would get from the app defined in
  [the AB test script](./examples/ab_test.py).
  
  ```javascript
    {
      "services": {
        "supa-dupa-model-ab-test": {
          "endpoint": "/supa-dupa-model/prediction",
          "status": "READY"
        }
      }
    }
  ```
  
**/<model name\>/prediction**: (Methods=`[POST]`):
  Each model added to the app will have an endpoint for accessing the model's predictions.
  The endpoint accepts `POST` requests with the input schema dependent on the model and
  resturns a JSON object with the following schema for batch predictions.
  
  ```javascript
    {
      "model_id": A unique identifier for the model,
      "model_version": A string identifying the model version,
      "predictions": [
         {"id": ..., "prediction": ...},
         ...
      ]
    }
  ```
  
  Single instance predictions return a JSON object with the schema
  
   ```javascript
    {
      "model_id": A unique identifier for the model,
      "model_version": A string identifying the model version,
      "predictions": {"id": ..., "prediction": ...}
    }
  ```
  
## Errors
If an error occurs while processing a request the user will receive a response with a non-200 status
code and JSON payload with the following keys

- "error": `string`. A simple name describing the error.
- "message": `string`. A more detailed error message.
- "traceback": `string`. The traceback of the `Exception` causing the error.
- "user_data": `object` or `null`. If the request contained a JSON payload it is returned to
  the user in this field. Otherwise, if no data was passed or the data was not valid JSON `null`
  is returned.
  
If the error resulted in a model context (during prediction, processing, etc.) the model context
data (model ID, version and any model meta data) described above will be present in the error object.

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
