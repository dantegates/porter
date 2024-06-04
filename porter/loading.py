"""Loading utilities."""


import io
import os
import tempfile

import joblib


_keras_extensions = (
    '.h5',
    '.keras',
)


def load_file(path, s3_access_key_id=None, s3_secret_access_key=None):
    """Load a file and return the result.

    Raises:
        ValueError: If ``path`` specifies an unknown file type or specifies an
            s3 resource but credentials are not provided.
    """
    extension = os.path.splitext(path)[-1]
    if path.startswith('s3://'):
        raise ValueError('S3 support has been deprecated')
    else:
        path_or_stream = path
    if extension == '.pkl':
        obj = load_pkl(path_or_stream)
    elif extension in _keras_extensions:
        # keras does not support loading a model from stream like joblib does.
        # as a workaround write the stream to a temporary file and load from
        # there.
        # See,
        # https://github.com/keras-team/keras/issues/9343
        if hasattr(path_or_stream, 'read'):
            with tempfile.NamedTemporaryFile() as tmp:
                with open(tmp.name, 'wb') as f:
                    # get buffer avoids copying the entire file contents
                    # like path_or_stream.read() would.
                    # https://docs.python.org/3/library/io.html#io.BytesIO.getbuffer
                    f.write(path_or_stream.getbuffer())
                obj = load_keras(tmp.name)
        else:
            obj = load_keras(path_or_stream)
    else:
        raise ValueError('unkown file type')
    return obj

def load_pkl(path):
    """Load and return a pickled object with ``joblib``."""
    model = joblib.load(path)
    return model

# on the reasonableness of imports inside a function, see
# https://stackoverflow.com/questions/3095071/in-python-what-happens-when-you-import-inside-of-a-function/3095167#3095167
def load_keras(path):
    """Load and return an object stored in h5 with ``tensorflow``."""
    import keras
    model = keras.models.load_model(path)
    return model
