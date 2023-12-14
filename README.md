# porter

| | |
| --- | --- |
| Testing | ![Unit Tests](https://github.com/dantegates/porter/actions/workflows/unit-tests.yml/badge.svg) |
| Documentation |  [![Documentation Status](https://readthedocs.org/projects/porter/badge/?version=latest)](https://porter.readthedocs.io/en/latest/?badge=latest) |
| Meta | [![License](https://img.shields.io/pypi/l/porter-schmorter)](./LICENSE) [![PyPI](https://img.shields.io/pypi/v/porter-schmorter)](https://pypi.org/project/porter-schmorter/) [![Python](https://img.shields.io/pypi/pyversions/porter-schmorter.svg)](https://pypi.org/project/porter-schmorter/) |




`porter` is a framework for data scientists who want to quickly and reliably deploy machine learning models as REST APIs. 

Simplicity is a core goal of this project. The following 6 lines of code are a fully functional example. While this should the most common use case, `porter` is also designed to be easily extended to cover the remaining cases not supported out of the box.

```python
from porter.datascience import WrappedModel
from porter.services import ModelApp, PredictionService

my_model = WrappedModel.from_file('my-model.pkl')
prediction_service = PredictionService(model=my_model, name='my-model', api_version='v1')

app = ModelApp([prediction_service])
app.run()
```

Features include:

* **Practical design**: suitable for projects ranging from proof-of-concept to production grade software.
* **Framework-agnostic design**: any object with a `predict()` method will do, which means `porter` plays nicely with [sklearn](https://scikit-learn.org/stable/), [keras](https://keras.io/backend/), or [xgboost](https://xgboost.readthedocs.io/en/latest/) models. Models that don't fit this pattern can be easily wrapped and used in ``porter``.
* **OpenAPI integration**: lightweight, Pythonic schema specifications support automatic validation of HTTP request data and generation of API documentation using Swagger.
* **Boiler plate reduction**: `porter` takes care of API logging and error handling out of the box, and supports streamlined model loading from `.pkl` and `.h5` files stored locally or on AWS S3.
* **Robust testing**: a comprehensive test suite ensures that you can use `porter` with confidence. Additionally, `porter` has been extensively field tested.

# Installation

`porter` can be installed with `pip` for `python3.9` and higher as follows:

```
pip install porter-schmorter  # because porter was taken
```

For more details, see [this page](https://porter.readthedocs.io/en/latest/installation.html).

# Documentation
For more information, see the [documentation](https://porter.readthedocs.org).
