import datetime
import json
import logging
import sys
import unittest
from unittest import mock

import io
import numpy as np
from porter.utils import (AppEncoder, JSONLogFormatter, NumpyEncoder,
                          PythonEncoder)

NOW = datetime.datetime.now()


class TestNumpyEncoder(unittest.TestCase):
    def test_default(self):
        encoder = NumpyEncoder()
        actual_type = type(encoder.default(np.int32(1)))
        expected_type = int
        self.assertIs(actual_type, expected_type)

        actual_type = type(encoder.default(np.float32(1)))
        expected_type = float
        self.assertIs(actual_type, expected_type)

        actual_type = type(encoder.default(np.array([[1]])))
        expected_type = list
        self.assertIs(actual_type, expected_type)

    def test_with_json_dumps(self):
        x = np.array([[np.float32(4.0)], [np.int32(0)]])
        actual = json.dumps(x, cls=NumpyEncoder)
        expected = '[[4.0], [0.0]]'
        self.assertEqual(actual, expected)


class TestPythonEncoder(unittest.TestCase):
    def test_default(self):
        encoder = PythonEncoder()
        actual_datetime = encoder.default(NOW)
        expected_datetime = NOW.isoformat()
        self.assertEqual(actual_datetime, expected_datetime)

        try:
            raise Exception('testing')
        except Exception as err:
            tb = sys.exc_info()[-1]
            actual_traceback = encoder.encode(tb)
        self.assertRegex(actual_traceback, 'File.*test_default.*raise.Exception')

        actual_type1 = encoder.default(ValueError('A value error'))
        expected_type1 = "ValueError('A value error')"
        self.assertEqual(actual_type1, expected_type1)

    def test_with_json_dumps(self):
        x = {'datetime': NOW, 'set': {1, 2}, 'exception': Exception('foo')}
        dump = json.dumps(x, cls=PythonEncoder)
        rec = json.loads(dump)
        self.assertEqual(rec['datetime'], NOW.isoformat())
        self.assertEqual(rec['set'], [1, 2])
        self.assertEqual(rec['exception'], "Exception('foo')")


class TestAppEncoder(unittest.TestCase):
    def test_default(self):
        encoder = AppEncoder()
        actual_type = type(encoder.default(np.int32(1)))
        expected_type = int
        self.assertIs(actual_type, expected_type)

        actual_type = type(encoder.default(np.float32(1)))
        expected_type = float
        self.assertIs(actual_type, expected_type)

        actual_type = type(encoder.default(np.array([[1]])))
        expected_type = list
        self.assertIs(actual_type, expected_type)

        actual_datetime = encoder.default(NOW)
        expected_datetime = NOW.isoformat()
        self.assertEqual(actual_datetime, expected_datetime)

        try:
            raise Exception('testing')
        except Exception as err:
            tb = sys.exc_info()[-1]
            actual_traceback = encoder.encode(tb)
        self.assertRegex(actual_traceback, 'File.*test_default.*raise.Exception')

        actual_type1 = encoder.default(ValueError('A value error'))
        expected_type1 = "ValueError('A value error')"
        self.assertEqual(actual_type1, expected_type1)

        actual_set = encoder.default({1, 2, 3})
        expected_set = [1, 2, 3]
        self.assertEqual(actual_set, expected_set)

    def test_json_dumps(self):
        encoder = AppEncoder()
        actual = json.dumps([
            {'array': np.array([[1, 2]]),
             'int': np.int32(10),
             'exception': ValueError('a value error')
            }
        ], cls=AppEncoder)
        expected = '[{"array": [[1, 2]], "int": 10, "exception": \"ValueError(\'a value error\')\"}]'
        self.assertEqual(actual, expected)

    def test_edge_case(self):
        encoder = AppEncoder()
        obj = object()
        # just make sure this doesn't raise an Exception
        actual = encoder.default(obj)


class TestJSONFormatterBig(unittest.TestCase):
    def setUp(self):
        logger = logging.getLogger('testlogger')
        logger.handlers = []
        logger.setLevel('INFO')
        sio = io.StringIO()
        console = logging.StreamHandler(sio)
        formatter = JSONLogFormatter('asctime', 'message', 'levelname')
        console.setFormatter(formatter)
        logger.addHandler(console)
        self.console, self.logger, self.sio = console, logger, sio

    def test_clean_dict(self):
        self.logger.info({'foo': 1, 'bar': 'baz'})
        actual = self.sio.getvalue()
        rec = json.loads(actual)
        self.assertRegex(rec['asctime'], f'{NOW.year}.*{NOW.month}.*{NOW.day}')
        self.assertEqual(rec['message'], {'foo': 1, 'bar': 'baz'})
        self.assertEqual(rec['levelname'], 'INFO')
        self.assertEqual(len(rec), 3)

    def test_clean_str(self):
        self.logger.info('whatsup %s', 'yo?')
        actual = self.sio.getvalue()
        rec = json.loads(actual)
        self.assertRegex(rec['asctime'], f'{NOW.year}.*{NOW.month}.*{NOW.day}')
        self.assertEqual(rec['message'], 'whatsup yo?')
        self.assertEqual(rec['levelname'], 'INFO')
        self.assertEqual(len(rec), 3)

    def test_exception(self):
        try:
            raise Exception('something bad')
        except Exception as err:
            self.logger.exception(err)
        actual = self.sio.getvalue()
        rec = json.loads(actual)
        self.assertRegex(rec['asctime'], f'{NOW.year}.*{NOW.month}.*{NOW.day}')
        self.assertEqual(rec['message'], 'something bad')
        self.assertEqual(rec['levelname'], 'ERROR')
        self.assertTrue(isinstance(rec['exc_info'], list))
        self.assertRegex(rec['exc_text'], "Traceback.*")


class TestJSONFormatterSmall(unittest.TestCase):
    @mock.patch('porter.utils.JSONLogFormatter._field_lookup_type', tuple)
    def test_format(self):
        record = mock.Mock(x=1, y=10, z=['foo', 10], exc_info=False, stack_info=False)
        formatter = JSONLogFormatter('x', 'y', 'z')
        formatter._field_lookup_type = tuple
        actual = formatter.format(record)
        expected = '{"x": 1, "y": 10, "z": ["foo", 10]}'
        self.assertEqual(actual, expected)

    @mock.patch('porter.utils.JSONLogFormatter._process_record')
    def test_format_small(self, mock__process_record):
        mock__process_record.return_value = {'foo': ['bar', 1], 'baz': 10}
        formatter = JSONLogFormatter()
        actual = formatter.format(mock.Mock())
        expected = '{"foo": ["bar", 1], "baz": 10}'
        self.assertEqual(actual, expected)

    def test__process_record_no_exc_no_stack(self):
        formatter = JSONLogFormatter('x', 'y', 'z')
        record = mock.Mock(x = 1, y=10, z=['foo', 10], exc_info=False, stack_info=False)
        actual = formatter._process_record(record)
        expected = {'x': 1, 'y': 10, 'z': ['foo', 10]}
        self.assertEqual(actual, expected)

    def test__process_record_with_exc_with_stack(self):
        formatter = JSONLogFormatter('x', 'y', 'z')
        try:
            raise Exception('foooooo')
        except Exception:
            record = mock.Mock(
                x=1, y=10, z=['foo', 10],
                exc_info=sys.exc_info(),
                exc_text=False,
                stack_info='stack info yo')
        actual = formatter._process_record(record)
        expected_kvps = {
            'x': 1, 'y': 10, 'z': ['foo', 10],
            'stack_info': 'stack info yo'
        }
        for k, v in expected_kvps.items():
            self.assertEqual(actual[k], v)
        self.assertRegex(repr(actual['exc_text']), 'Traceback.*raise Exception')


if __name__ == '__main__':
    unittest.main()
