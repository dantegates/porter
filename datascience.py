import os


# on the acceptability of imports inside a function, see
# https://stackoverflow.com/questions/3095071/in-python-what-happens-when-you-import-inside-of-a-function/3095167#3095167
def load_pkl(path):
    from sklearn.externals import joblib
    model = joblib.load(path)
    return model

def load_h5(path):
    import keras
    model = keras.models.load_model(path)
    return model


class Model:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.predict(X)

    def get_feature_names(self):
        raise NotImplementedError

    def get_schema(self):
        raise NotImplementedError

    @classmethod
    def from_file(cls, path):
        extension = os.path.splitext(path)[-1]
        if extension == '.pkl':
            model = load_pkl(path)
        elif extension == '.h5':
            model = load_h5(path)
        else:
            raise Exception('file type unkown')
        return cls(model)
