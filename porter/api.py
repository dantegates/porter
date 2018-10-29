"""Light wrappers around `flask` and `requests`."""


import json

import flask


def request_method():
    return flask.request.method


def request_json(*args, **kwargs):
    return flask.request.get_json(*args, **kwargs)


def jsonify(*args, **kwargs):
    return flask.jsonify(*args, **kwargs)


App = flask.Flask


def post(*args, **kwargs):
    # requests should be considered an optional dependency.
    # for additional details on this pattern see the loading module.
    import requests
    return requests.post(*args, **kwargs)


def get(*args, **kwargs):
    # requests should be considered an optional dependency.
    # for additional details on this pattern see the loading module.
    import requests as rq
    return rq.get(*args, **kwargs)
