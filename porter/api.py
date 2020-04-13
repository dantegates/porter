"""Light wrappers around ``flask`` and ``requests``."""


import functools
import json
import uuid

import flask


def request_method():
    """Return the HTTP method of the current request, e.g. 'GET', 'POST', etc."""
    return flask.request.method


def request_json(*args, **kwargs):
    """Return the JSON from the current request."""
    return flask.request.get_json(*args, **kwargs)


def jsonify(data, *args, **kwargs):
    """'Jsonify' a Python object into something an instance of :class:`App` can return
    to the user.
    """
    jsonified = flask.jsonify(data, *args, **kwargs)
    jsonified.raw_data = data
    return jsonified


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
