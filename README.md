# porter
porter is a framework for exposing machine learning models via REST APIs.

# Usage
The basic workflow for building a model service is as follows

1. Instantiate an instance of `porter.services.ModelApp`. This is simply an abstraction for
  the REST app that will expose your models.
2. Define model classes for each service you want to add to the app. A single service consists of
  a model that satisfies the `porter.datascience.BaseModel` interface. Additionally you can define
  processor classes (objects implementing the `porter.datascience.BaseProcessor` interface`) for
  pre/post processing of model input/output respectively.
3. Instantiate `porter.services.PredictionServiceConfig` with the appropriate arguments once for
  each model you would expose through the app.
4. Pass the config objects from 3. to the `add_serivce` method of your `ModelApp` instance.
5. Call the `run` method of your `ModelApp` instance. Your model is now live!

See [example.py](./example.py) for an illustrative but non-functional example.
