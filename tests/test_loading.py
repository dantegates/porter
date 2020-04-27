import os
import tempfile
import unittest

import boto3
import numpy as np
import sklearn.linear_model
import tensorflow as tf
from porter import loading
from sklearn.externals import joblib


class BaseTestLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s3_access_key_id = os.environ['PORTER_S3_ACCESS_KEY_ID']
        cls.s3_secret_access_key = os.environ['PORTER_S3_SECRET_ACCESS_KEY']
        cls.bucket = os.environ['PORTER_S3_BUCKET_TEST']

    def write_to_s3(self, obj, bucket, key):
        s3 = boto3.client('s3',
                          aws_access_key_id=self.s3_access_key_id,
                          aws_secret_access_key=self.s3_secret_access_key)
        s3.upload_file(obj, bucket, key, ExtraArgs={'ServerSideEncryption': 'AES256'})


class TestLoadingSklearn(BaseTestLoading):
    @classmethod
    def setUpClass(cls):
        cls.X = np.random.rand(10, 20)
        cls.y = np.random.randint(1, 10, size=10)
        cls.model = sklearn.linear_model.SGDRegressor(max_iter=10)
        cls.model.fit(cls.X, cls.y)
        cls.predictions = cls.model.predict(cls.X)
        super().setUpClass()

    def test_load_pkl(self):
        with tempfile.NamedTemporaryFile(suffix='.pkl') as tmp:
            joblib.dump(self.model, tmp.name)
            loaded_model = loading.load_pkl(tmp.name)
        actual_predictions = loaded_model.predict(self.X)
        expected_predictions = self.predictions
        self.assertTrue(np.allclose(actual_predictions, expected_predictions))

    def test_load_file_pkl(self):
        with tempfile.NamedTemporaryFile(suffix='.pkl') as tmp:
            joblib.dump(self.model, tmp.name)
            loaded_model = loading.load_file(tmp.name)
        actual_predictions = loaded_model.predict(self.X)
        expected_predictions = self.predictions
        self.assertTrue(np.allclose(actual_predictions, expected_predictions))

    def test_load_file_s3(self):
        key = 'data-science/porter/tests/sklearn_model.pkl'
        s3_path = 's3://%s/%s' % (self.bucket, key)
        with tempfile.NamedTemporaryFile(suffix='.pkl') as tmp:
            joblib.dump(self.model, tmp.name)
            self.write_to_s3(tmp.name, self.bucket, key)
        loaded_model = loading.load_file(s3_path,
                s3_access_key_id=self.s3_access_key_id,
                s3_secret_access_key=self.s3_secret_access_key)
        actual_predictions = loaded_model.predict(self.X)
        expected_predictions = self.model.predict(self.X)
        self.assertTrue(np.allclose(actual_predictions, expected_predictions))


class TestLoadingKeras(BaseTestLoading):
    @classmethod
    def setUpClass(cls):
        cls.X = np.random.rand(10, 20)
        cls.y = np.random.randint(1, 10, size=10)
        cls.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(20, input_shape=(20,)),
            tf.keras.layers.Dense(1)
        ])
        cls.model.compile(loss='mean_squared_error', optimizer='sgd')
        cls.model.fit(cls.X, cls.y, verbose=0)
        cls.predictions = cls.model.predict(cls.X)
        super().setUpClass()

    def test_load_h5(self):
        with tempfile.NamedTemporaryFile(suffix='.h5') as tmp:
            tf.keras.models.save_model(self.model, tmp.name)
            loaded_model = loading.load_h5(tmp.name)
        actual_predictions = loaded_model.predict(self.X)
        expected_predictions = self.predictions
        self.assertTrue(np.allclose(actual_predictions, expected_predictions))

    def test_load_file_h5(self):
        with tempfile.NamedTemporaryFile(suffix='.h5') as tmp:
            tf.keras.models.save_model(self.model, tmp.name)
            loaded_model = loading.load_file(tmp.name)
        actual_predictions = loaded_model.predict(self.X)
        expected_predictions = self.predictions
        self.assertTrue(np.allclose(actual_predictions, expected_predictions))

    def test_load_file_s3(self):
        key = 'data-science/porter/tests/sklearn_model.h5'
        s3_path = 's3://%s/%s' % (self.bucket, key)
        with tempfile.NamedTemporaryFile(suffix='.h5') as tmp:
            tf.keras.models.save_model(self.model, tmp.name)
            self.write_to_s3(tmp.name, self.bucket, key)
        loaded_model = loading.load_file(s3_path,
                s3_access_key_id=self.s3_access_key_id,
                s3_secret_access_key=self.s3_secret_access_key)
        actual_predictions = loaded_model.predict(self.X)
        expected_predictions = self.model.predict(self.X)
        self.assertTrue(np.allclose(actual_predictions, expected_predictions))


class TestLoadingS3(BaseTestLoading):
    def test_load_s3(self):
        content = 'foo bar baz'
        with tempfile.NamedTemporaryFile() as tmp:
            with open(tmp.name, 'w') as f:
                f.write(content)
            key = 'data-science/porter/tests/arbitrary_content'
            self.write_to_s3(tmp.name, self.bucket, key)
        actual_content = loading.load_s3(
            's3://%s/%s' % (self.bucket, key),
            s3_access_key_id=self.s3_access_key_id,
            s3_secret_access_key=self.s3_secret_access_key)
        actual = actual_content.read()
        expected = bytes(content, 'utf-8')
        self.assertEqual(actual, expected)

    def test_load_file_s3_fail_missing_key(self):
        self.bucket = os.environ['PORTER_S3_BUCKET_TEST']
        with self.assertRaisesRegex(Exception, r'An error occurred \(404\)'):
            loading.load_file('s3://%s/this/does/not/exist' % self.bucket,
                s3_access_key_id=self.s3_access_key_id,
                s3_secret_access_key=self.s3_secret_access_key)

    def test_load_file_s3_fail_missing_bucket(self):
        with self.assertRaisesRegex(Exception, r'An error occurred \(403\)'):
            loading.load_file('s3://invalid-bucket/this/does/not/exist',
                s3_access_key_id=self.s3_access_key_id,
                s3_secret_access_key=self.s3_secret_access_key)

    def test_split_s3_path(self):
        path = 's3://my-bucket/some/key'
        actual_bucket, actual_key = loading.split_s3_path(path)
        expected_bucket, expected_key = 'my-bucket', 'some/key'
        self.assertEqual(actual_bucket, expected_bucket)
        self.assertEqual(actual_key, expected_key)


if __name__ == '__main__':
    unittest.main()
