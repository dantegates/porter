"""Light wrappers around `flask` and `requests`."""


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
    """'Jsonify' a Python object into something an instance of `App` can return
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


def cache_during_request(f):
    """A decorator that can be applied to functions to cache results during
    the lifetime of a request.

    Note that if the function has already been called during a request they
    are ignored during subsequent calls and the original result is returned.
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        f_id = str(id(f))
        res = getattr(flask.g, f_id, None)
        if res is None:
            res = f(*args, **kwargs)
            setattr(flask.g, f_id, res)
        return res
    return wrapper


App = flask.Flask


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
    """Return True if `url` is valid and False otherwise.

    Roughly speaking, a valid URL is a URL containing sufficient information
    for `post()` and `get()` to send requests - whether or not the URL actually
    exists.
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
