import os


# on the reasonableness of imports inside a function, see
# https://stackoverflow.com/questions/3095071/in-python-what-happens-when-you-import-inside-of-a-function/3095167#3095167
def load_pkl(path):
    """Load and return a pickled object with `joblib`."""
    from sklearn.externals import joblib
    model = joblib.load(path)
    return model

def load_h5(path):
    """Load and return an object stored in h5 with `keras`."""
    import keras
    model = keras.models.load_model(path)
    return model

def load_file(path):
    """Load a file and return the result."""
    extension = os.path.splitext(path)[-1]
    if extension == '.pkl':
        obj = load_pkl(path)
    elif extension == '.h5':
        obj = load_h5(path)
    else:
        raise Exception('unkown file type')
    return obj


class BaseModel(object):
    """Class defining the model interface required by
        `porter.services.ModelApp.add_service`."""
    def predict(self, X):
        raise NotImplementedError(
            '%s must implement .predict()' % self.__class__.__name__)


class BaseProcessor(object):
    """Class defining the [pre|post]processor interface required by
        `porter.services.ModelApp.add_service`."""
    def process(self, X):
        raise NotImplementedError(
            '%s must implement .process()' % self.__class__.__name__)


class WrappedModel(BaseModel):
    """A convenience class that exposes a model persisted to disk with the
    `BaseModel` interface.
    """
    def __init__(self, model):
        self.model = model
        super(WrappedModel, self).__init__()

    def predict(self, X):
        return self.model.predict(X)

    @classmethod
    def from_file(cls, path, *args, **kwargs):
        model = load_file(path)
        return cls(model, *args, **kwargs)


class WrappedTransformer(BaseProcessor):
    """A convenience class that exposes a transformer persisted to disk with
    the `BaseProcessor` interface.
    """
    def __init__(self, transformer):
        self.transformer = transformer
        super(WrappedTransformer, self).__init__()

    def process(self, X):
        return self.transformer.transform(X)

    @classmethod
    def from_file(cls, path, *args, **kwargs):
        transformer = load_file(path)
        return cls(transformer, *args, **kwargs)
