# porter
porter is a framework for exposing machine learning models via REST APIs.

# Installation
`porter` can be installed with `pip` as follows

```shell
pip install git+https://github.com/CadentTech/porter#egg=porter
```

Note that without `#egg=porter` on the end of the url `pip freeze` will output `porter==<version>`
rather than `-e git+https://...` as typically
desired.

If you want to install `porter` from a specific commit or tag, e.g. tag `1.0.0` simply and 
`@<commit-or-tag>` immediately before `#egg=porter`.

```shell
pip install git+https://github.com/CadentTech/porter@1.0.0#egg=porter
```

For more details on this topic see [here](https://codeinthehole.com/tips/using-pip-and-requirementstxt-to-install-from-the-head-of-a-github-branch/)

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
3. Instantiate `porter.services.PredictionServiceConfig` with the appropriate arguments once for
  each model you would expose through the app.
4. Pass the config objects from 3. to the `add_serivce` method of your `ModelApp` instance.
5. Call the `run` method of your `ModelApp` instance. Your model is now live!

See this [example script](./examples/example.py) for an (almost functional) example.

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
