import os
_resource_directory = os.path.join(
    os.path.dirname(__file__),
    '..',
    'resources')
SKLEARN_MODEL_PATH = os.path.join(_resource_directory, 'sklearn-model.pkl')
KERAS_MODEL_PATH = os.path.join(_resource_directory, 'keras-model.h5')
SKLEARN_VALIDATE_PATH = os.path.join(_resource_directory, 'sklearn-val.json')
KERAS_VALIDATE_PATH = os.path.join(_resource_directory, 'sklearn-val.json')
