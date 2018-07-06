import argparse
import functools
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests as rq


def stress_tests(fn=None, tests=[]):
    """Register or return all registered stress tests."""
    if fn is not None:
        tests.append(fn)
        return fn
    return tests


class Timer:
    """Simple timer class."""
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *exc):
        self.stop = time.time()

    @property
    def elapsed(self):
        return self.stop - self.start


def init_cli():
    """Build the CLI, parse the CLI arguments and return as dict."""
    cli = argparse.ArgumentParser(description='stress test model APIs')
    cli.add_argument('--host', type=str, default='192.168.99.100')
    cli.add_argument('--port', type=str, default='8000')
    cli.add_argument('--model-name', type=str, default=None, required=True)
    cli.add_argument('--max-requests', type=int, default=201)
    cli.add_argument('--max-requests-skip-by', type=int, default=40)
    cli.add_argument('--batch-request-size', type=int, default=250)
    cli.add_argument('--example-input', type=str, default='example_api_input.json')
    cli.add_argument('--example-bad-input', type=str, default='example_api_bad_input.json')
    args = cli.parse_args()
    args.root_url = f'http://{args.host}:{args.port}'
    args.prediction_url = f'{args.root_url}/{args.model_name}/prediction'
    args.alive_url = f'{args.root_url}/-/alive'
    args.ready_url = f'{args.root_url}/-/ready'
    with open(args.example_input) as f:
        args.data = f.read()
    with open(args.example_bad_input) as f:
        args.bad_data = f.read()
    return dict(vars(args).items())


@stress_tests
def health_endpoints(root_url, alive_url, ready_url, **kwargs):
    """Sending GET requests to health endpoints."""
    get_alive = f'GET {alive_url.replace(root_url, "")}'
    get_alive_resp = rq.get(alive_url)
    get_ready = f'GET {ready_url.replace(root_url, "")}'
    get_ready_resp = rq.get(ready_url)
    print(f'{get_alive}, status code = {get_alive_resp.status_code}')
    print(f'{get_alive}, payload = {get_alive_resp.json()}')
    print(f'{get_ready}, status code = {get_ready_resp.status_code}')
    print(f'{get_ready}, payload = {get_ready_resp.json()}')


@stress_tests
def bad_requests(root_url, prediction_url, bad_data, **kwargs):
    """Sending a bunch of bad requests to the app."""
    badnesses = [
        functools.partial(rq.get, prediction_url),
        functools.partial(rq.post, prediction_url, data=bad_data),
        functools.partial(rq.get, f'{root_url}/doesnotexist'),]
    for badness in badnesses:
        response = badness()


@stress_tests
def hammer(prediction_url, data, max_requests, max_requests_skip_by, **kwargs):
    """Sending lots of small requests to the app concurrently."""
    def get_prediction(url, data):
        response = rq.post(url, data=data)
        return response.status_code
    for n_requests in range(1, max_requests+1, max_requests_skip_by):
        # number of CPUs * 5, see docs
        with ThreadPoolExecutor(max_workers=None) as executor:
            with Timer() as timer:
                futures = [executor.submit(get_prediction, prediction_url, data)
                           for _ in range(n_requests)]
                status_codes = Counter([future.result()
                                        for future in as_completed(futures)])
            message = '\n'.join([
                f'n_requests={n_requests}',
                f'status_codes={status_codes}',
                f'completed in {timer.elapsed} secs.',])
            print(message)


@stress_tests
def mallet(prediction_url, data, max_requests, max_requests_skip_by,
           batch_request_size, **kwargs):
    """Sending lots of big(ger) requests to the app concurrently."""
    # inflate the data
    data = json.dumps([json.loads(data)[0]
                       for _ in range(batch_request_size)])
    def get_prediction(url, data):
        response = rq.post(url, data=data)
        return response.status_code
    for n_requests in range(1, max_requests+1, max_requests_skip_by):
        # number of CPUs * 5, see docs
        with ThreadPoolExecutor(max_workers=None) as executor:
            with Timer() as timer:
                futures = [executor.submit(get_prediction, prediction_url, data)
                           for _ in range(n_requests)]
                status_codes = Counter([future.result()
                                        for future in as_completed(futures)])
            message = '\n'.join([
                f'n_requests={n_requests}',
                f'status_codes={status_codes}',
                f'completed in {timer.elapsed} secs.',])
            print(message)


def describe_test(test):
    title =f'running {test.__name__}'
    descr = test.__doc__
    underline = "-" * len(title)
    print(f'\n{title}\n{descr}\n{underline}')


if __name__ == '__main__':
    tests = stress_tests()
    kwargs = init_cli()
    for test in stress_tests():
        describe_test(test)
        test(**kwargs)
