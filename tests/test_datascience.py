import unittest
from unittest import mock

from porter.datascience import (BaseModel, BasePostProcessor, BasePreProcessor,
                                WrappedModel, WrappedTransformer)


class TestBaseModel(unittest.TestCase):
    def test_abc(self):
        class A(BaseModel): pass
        with self.assertRaises(TypeError):
            A()


class TestWrappedModel(unittest.TestCase):
    def test_predict(self):
        mock_model = mock.Mock()
        mock_model.predict = lambda x: x+1
        model = WrappedModel(mock_model)
        actual = model.predict(1)
        expected = 2
        self.assertEqual(actual, expected)

    def test_from_file_sklearn(self):
        pass

    def test_from_file_keras(self):
        pass

    def test_model_validation(self):
        class A:
            pass
        class B:
            predict = 42
        class C:
            def predict(x):
                return x + 1
        msg = 'model must have a .predict\(\) method'
        with self.assertRaisesRegex(TypeError, msg):
            WrappedModel(A())
        with self.assertRaisesRegex(TypeError, msg):
            WrappedModel(B())
        WrappedModel(C())


class TestBasePreProcessor(unittest.TestCase):
    def test_abc(self):
        class A(BasePreProcessor): pass
        with self.assertRaises(TypeError):
            A()


class TestBasePostProcessor(unittest.TestCase):
    def test_abc(self):
        class A(BasePostProcessor): pass
        with self.assertRaises(TypeError):
            A()


class TestWrappedTransformer(unittest.TestCase):
    def test_transform(self):
        class MockTransformer:
            def transform(self, X):
                return X + 1
        processor = WrappedTransformer(MockTransformer())
        actual = processor.process(1)
        expected = 2
        self.assertEqual(actual, expected)

    def test_transformer_validation(self):
        class A:
            pass
        class B:
            transform = 42
        class C:
            def transform(x):
                return x + 1
        msg = 'transformer must have a .transform\(\) method'
        with self.assertRaisesRegex(TypeError, msg):
            WrappedTransformer(A())
        with self.assertRaisesRegex(TypeError, msg):
            WrappedTransformer(B())
        WrappedTransformer(C())

if __name__ == '__main__':
    unittest.main()
