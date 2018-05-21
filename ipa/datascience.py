import os


# on the reasonableness of imports inside a function, see
# https://stackoverflow.com/questions/3095071/in-python-what-happens-when-you-import-inside-of-a-function/3095167#3095167
def load_pkl(path):
    from sklearn.externals import joblib
    model = joblib.load(path)
    return model

def load_h5(path):
    import keras
    model = keras.models.load_model(path)
    return model

def load_file(path):
    extension = os.path.splitext(path)[-1]
    if extension == '.pkl':
        obj = load_pkl(path)
    elif extension == '.h5':
        obj = load_h5(path)
    else:
        raise Exception('unkown file type')
    return obj


class BaseModel(object):
    def __init__(self, name, id):
        self.name = name
        self.id = id

    def predict(self, X):
        raise NotImplementedError(
            '%s must implement .predict()' % self.__class__.__name__)


class BaseFeatureEngineer(object):
    def transform(self):
        raise NotImplementedError(
            '%s must implement .transform()' % self.__class__.__name__)


class WrappedModel(BaseModel):
    def __init__(self, model, name, id):
        self.model = model
        super(WrappedModel, self).__init__(name, id)

    def predict(self, X):
        return self.model.predict(X)

    @classmethod
    def from_file(cls, path, *args, **kwargs):
        model = load_file(path)
        return cls(model, *args, **kwargs)


class WrappedFeatureEngineer(BaseFeatureEngineer):
    def __init__(self, transformer):
        self.transformer = transformer
        super(WrappedFeatureEngineer, self).__init__()

    def transform(self, X):
        return self.transformer.transform(X)

    @classmethod
    def from_file(cls, path, *args, **kwargs):
        transformer = load_file(path)
        return cls(transformer, *args, **kwargs)
