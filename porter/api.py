"""Light wrappers around ``flask`` and ``requests``."""


import functools
import gzip
import io
import json
import uuid

import flask
import werkzeug.exceptions as werkzeug_exc

from . import config as cf


def request_method():
    """Return the HTTP method of the current request, e.g. 'GET', 'POST', etc."""
    return flask.request.method


def request_json(silent=False):
    """Return the JSON from the current request.

    Args:
        silent (bool): Silence parsing errors and return None instead.
    """
    request = flask.request
    encoding = str(request.content_encoding).lower()
    bad_request = werkzeug_exc.BadRequest(
        'The browser (or proxy) sent a request that this server could not understand.')
    if encoding == 'gzip':
        try:
            data = json.loads(gzip.decompress(request.get_data()).decode('utf-8'))
        except:
            if not silent:
                raise bad_request
    elif encoding in ('identity', 'none'):
        try:
            data = request.get_json(force=True)
        except:
            if not silent:
                raise bad_request
    else:
        if not silent:
            # TODO: check error message wording
            raise werkzeug_exc.UnsupportedMediaType(f'unsupported encoding: "{encoding}"')
    return data


def jsonify(data, *, status_code):
    """'Jsonify' a Python object into something an instance of :class:`App` can return
    to the user.
    """
    jsonified = flask.jsonify(data)
    jsonified.status_code = status_code
    jsonified.raw_data = data
    jsonified.is_gzipped = False
    if status_code == 200:
        encode_response(jsonified)
    return jsonified


def _gzip_response(response):
    response.direct_passthrough = False
    response.data = gzip.compress(response.data)

    response.headers['Content-Encoding'] = 'gzip'
    response.headers['Vary'] = 'Accept-Encoding'
    response.headers['Content-Length'] = len(response.data)
    response.is_gzipped = True

def encode_response(response):
    """Encode response if a supported value of ``Accept-Encoding`` is passed."""
    # See https://kb.sites.apiit.edu.my/knowledge-base/how-to-gzip-response-in-flask/

    # Short circuit if no encoding support enabled
    if not cf.support_gzip:
        return

    accept_encoding = flask.request.headers.get('Accept-Encoding', '').lower()

    if 'gzip' in accept_encoding and cf.support_gzip:
        _gzip_response(response)
    else:
        # If the client requests an unsupported encoding,
        # it may be appropriate to respond with 406 Not Acceptable.
        # However, it is probably preferrable to simply respond with
        # "identity" encoding instead.
        # https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/406
        pass
    return response


def request_id():
    """Return a "unique" ID for the current request."""
    # http://flask.pocoo.org/docs/dev/tutorial/dbcon/
    if not hasattr(flask.g, 'request_id'):
        flask.g.request_id = uuid.uuid4().hex
    return flask.g.request_id

def set_model_context(service):
    """Register a model on the request context.

    Args:
        service (:class:`porter.sevices.BaseService`)
    """
    # http://flask.pocoo.org/docs/dev/tutorial/dbcon/
    flask.g.model_context = service

def get_model_context():
    """Returns :class:`porter.sevices.BaseService` or None"""
    # http://flask.pocoo.org/docs/dev/tutorial/dbcon/
    return getattr(flask.g, 'model_context', None)


App = flask.Flask
"""alias of ``flask.app.Flask``."""

def post(*args, data, **kwargs):
    # requests should be considered an optional dependency.
    # for additional details on this pattern see the loading module.
    import requests
    return requests.post(*args, data=json.dumps(data), **kwargs)


def get(*args, **kwargs):
    # requests should be considered an optional dependency.
    # for additional details on this pattern see the loading module.
    import requests
    return requests.get(*args, **kwargs)


def validate_url(url):
    """Return True if ``url`` is valid and False otherwise.

    Roughly speaking, a valid URL is a URL containing sufficient information
    for :meth:`post()` and :meth:`get()` to send requests - whether or not the URL
    actually exists.
    """
    from urllib3.util import parse_url
    # basically following the implementation here
    # https://github.com/requests/requests/blob/75bdc998e2d430a35d869b2abf1779bd0d34890e/requests/models.py#L378
    try:
        parts = parse_url(url)
    except Exception:
        is_valid = False
    is_valid = parts.scheme and parts.host
    return is_valid

