"""Light wrappers around `flask` and `requests`."""


import json

import flask


def request_method():
    """Return the HTTP method of the current request, e.g. 'GET', 'POST', etc."""
    return flask.request.method


def request_json(*args, **kwargs):
    """Return the JSON from the current request."""
    return flask.request.get_json(*args, **kwargs)


def jsonify(*args, **kwargs):
    """'Jsonify' a Python object into something an instance of `App` can return
    to the user.
    """
    return flask.jsonify(*args, **kwargs)


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
