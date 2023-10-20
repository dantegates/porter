import datetime
import io
import json
import logging
import traceback
from inspect import istraceback

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """A JSON encoder that handles ``numpy`` data types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class PythonEncoder(json.JSONEncoder):
    """
    A JSON encoder that extends ``json.JSONEncoder`` to handle additional Python
    types.
    """

    def default(self, obj):
        if isinstance(obj, (datetime.date, datetime.datetime, datetime.time)):
            return obj.isoformat()
        elif isinstance(obj, set):
            return list(obj)
        elif istraceback(obj):
            with io.StringIO() as string_io:
                traceback.print_tb(obj, file=string_io)
                return string_io.getvalue()
        elif isinstance(obj, Exception):
            return repr(obj)
        return super().default(obj)


class AppEncoder(NumpyEncoder, PythonEncoder):
    """A JSON encoder that handles ``numpy`` and python data types."""

    def default(self, obj):
        try:
            return super().default(obj)
        except TypeError:
            try:
                return str(obj)
            except Exception:
                pass


class JSONLogFormatter(logging.Formatter):
    """A JSON formatter for logs.

    Usage:

        >>> logger = logging.getLogger(__name__)
        >>> logger.setLevel('INFO')
        >>> console = logging.StreamHandler()
        >>> formatter = JSONLogFormatter('asctime', 'message', 'levelname')
        >>> console.setFormatter(formatter)
        >>> logger.addHandler(console)
        >>> logger.info({'something': 'interesting'})
        {"message": {"something": "interesting"}, "levelname": "INFO",
         "asctime": "2018-07-05 11:48:54,248"}

    Any attribute of ``logging.LogRecord`` can be specified to log. See

        https://docs.python.org/3/library/logging.html#logging.LogRecord

    Exception information does not need to be specified. If there was an
    exception, that information is added automatically to the log.

        >>> try:
        >>>     raise Exception('something bad')
        >>> except Exception as err:
        >>>     logger.exception(err)
        {"exc_text": "...", "exc_info": "...", "levelname": "ERROR", 
         "asctime": "2018-07-05 11:51:26,179", "message": "something bad"}

    Args:
        fields (list of str): List of fields to include in log. This can be
            any attribute of a ``logging.LogRecord`` object plus "asctime" and
            "message". For a list of ``logging.LogRecord`` attributes.
        indent (int or None): The indentation level. Values are the same as
            ``json.dump``.
        encoder (object): A ``json.JSONEncoder`` subclass. It is recommended to
            use ``Encoder`` or a subclass thereof if you need additional
            handling so that ``Exception``'s and ``datetime``'s are properly
            handled.
        **kwargs: Additional keyword arguments to be passed to
            ``logging.Formatter.__init__``.
    """

    # tests can override this value as a `tuple` to preserve order.
    _field_lookup_type = set

    def __init__(self, *fields, indent=None, encoder=None, **kwargs):
        self.fields = self._field_lookup_type(fields)
        self.encoder = AppEncoder if encoder is None else encoder
        self.indent = indent
        super().__init__(**kwargs)

    def format(self, record):
        record = self._process_record(record)
        return json.dumps(record, cls=self.encoder, indent=self.indent)

    def _process_record(self, record):
        # following python implementation
        # https://github.com/python/cpython/blob/f55a818954212e8e6c97e3d66cf1478120a3220f/Lib/logging/__init__.py#L563
        fields = self._field_lookup_type(self.fields)
        if 'message' in fields:
            # skip default implementation which converts `record.msg` to a
            # `dict` and then performs string formatting when `record.msg` is
            # a `dict`.
            if isinstance(record.msg, dict):
                record.message = record.msg
            else:
                record.message = record.getMessage()
        if 'asctime' in fields:
            record.asctime = self.formatTime(record, self.datefmt)
        if record.exc_info:
            fields.add('exc_info')
            # Cache the traceback text to avoid converting it multiple times
            # (it's constant anyway)
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
            fields.add('exc_text')
        if record.stack_info:
            fields.add('stack_info')
        return {field: getattr(record, field, None) for field in fields}


def object_constants(obj):
    return [(attr, val) for attr, val in vars(obj).items()
            if not attr.startswith('_') and attr.isupper()]

# keep this reference for backwards compatibility
JSONFormatter = JSONLogFormatter
