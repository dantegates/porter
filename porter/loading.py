"""Loading utilities."""


import os


def load_file(path):
    """Load a file and return the result."""
    extension = os.path.splitext(path)[-1]
    if path.startswith('s3://'):
        s3_access_key_id = os.environ.get('PORTER_S3_ACCESS_KEY_ID')
        s3_secret_access_key = os.environ.get('PORTER_S3_SECRET_ACCESS_KEY')
        path_or_stream = load_s3(path, s3_access_key_id, s3_secret_access_key)
    if extension == '.pkl':
        obj = load_pkl(path_or_stream)
    elif extension == '.h5':
        obj = load_h5(path_or_stream)
    else:
        raise Exception('unkown file type')
    return obj

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

def load_s3(path, s3_access_key_id, s3_secret_access_key):
    import boto3
    import botocore
    s3_client = boto3.client(
        's3',
        aws_access_key_id=s3_access_key_id,
        aws_secret_access_key=s3_secret_access_key)
    bucket, key = split_s3_path(path)
    try:
        stream = s3_client.download_fileobj(bucket, key)
    except botocore.exceptions.ClientError as err:
        if err.response['Error']['Code'] == "404":  # not found
            raise RuntimeError(
                'The object bucket=%s, key=%s does not exist in S3'
                % (bucket, key))
        else:
            raise err
    return stream

def split_s3_path(path):
    if path.startswith('s3://'):
        _, path = path.split('s3://')
    bucket, _, key = path.partition('/')
    return bucket, key
