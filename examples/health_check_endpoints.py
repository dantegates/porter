import json
import os
import sys
import threading
import urllib.error
import urllib.request

from porter.services import ModelApp, PredictionServiceConfig

service_config_1 = PredictionServiceConfig(
    model=None,
    name='a-model',
    version='0.0.0'
)

service_config_2 = PredictionServiceConfig(
    model=None,
    name='yet-another-model',
    version='1.0.0'
)

service_config_3 = PredictionServiceConfig(
    model=None,
    name='yet-another-yet-another-model',
    version='1.0.0-alpha',
    meta={'arbitrary details': 'about the model'}
)

model_app = ModelApp()
model_app.add_services(service_config_1, service_config_2, service_config_3)


def get(url):
    with urllib.request.urlopen(url) as f:
        return f.read()


def run_app(model_app):
    t = threading.Thread(target=model_app.run, daemon=True)
    t.start()


class Shhh:
    """Silence flask logging."""

    def __init__(self):
        self.devnull = open(os.devnull, 'w')
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def __enter__(self):
        sys.stdout = self.devnull
        sys.stderr = self.devnull

    def __exit__(self, *exc):
        sys.stdout = self.stdout
        sys.stderr = self.stderr


if __name__ == '__main__':
    with Shhh():
        run_app(model_app)
        while True:
            try:
                alive_resp = json.loads(get('http://localhost:5000/-/alive').decode('utf-8'))
                ready_resp = json.loads(get('http://localhost:5000/-/alive').decode('utf-8'))
                break
            except urllib.error.URLError:
                pass
    print('GET /-/alive')
    print(json.dumps(alive_resp, indent=4))
    print('GET /-/ready')
    print(json.dumps(ready_resp, indent=4))
