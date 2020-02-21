"""Loading utilities."""


import io
import os
import tempfile

from . import exceptions as exc


def load_file(path, s3_access_key_id=None, s3_secret_access_key=None):
    """Load a file and return the result."""
    extension = os.path.splitext(path)[-1]
    if path.startswith('s3://'):
        if s3_access_key_id is None or s3_secret_access_key is None:
            raise exc.PorterError(
                's3_access_key_id and s3_secret_access_key cannot be None '
                'when loading a resource from S3.')
        path_or_stream = load_s3(path, s3_access_key_id, s3_secret_access_key)
    else:
        path_or_stream = path
    if extension == '.pkl':
        obj = load_pkl(path_or_stream)
    elif extension == '.h5':
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
                obj = load_h5(tmp.name)
        else:
            obj = load_h5(path_or_stream)
    else:
        raise exc.PorterError('unkown file type')
    return obj

# on the reasonableness of imports inside a function, see
# https://stackoverflow.com/questions/3095071/in-python-what-happens-when-you-import-inside-of-a-function/3095167#3095167
def load_pkl(path):
    """Load and return a pickled object with ``joblib``."""
    from sklearn.externals import joblib
    model = joblib.load(path)
    return model

def load_h5(path):
    """Load and return an object stored in h5 with ``tensorflow``."""
    import tensorflow as tf
    model = tf.keras.models.load_model(path)
    return model

def load_s3(path, s3_access_key_id, s3_secret_access_key):
    import boto3
    import botocore
    s3_client = boto3.client(
        's3',
        aws_access_key_id=s3_access_key_id,
        aws_secret_access_key=s3_secret_access_key)
    bucket, key = split_s3_path(path)
    try:
        stream = io.BytesIO()
        _ = s3_client.download_fileobj(bucket, key, stream)
    except botocore.exceptions.ClientError as err:
        if err.response['Error']['Code'] == "404":  # not found
            raise exc.PorterError(
                'tried to read an object in S3 that does not exist: '
                f'bucket={bucket}, key={key}')
        else:
            raise err
    stream.seek(0)
    return stream

def split_s3_path(path):
    if path.startswith('s3://'):
        _, path = path.split('s3://')
    bucket, _, key = path.partition('/')
    return bucket, key
