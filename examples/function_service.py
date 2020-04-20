from porter.services import BaseService, ModelApp
import porter.api as porter_api
from porter import schemas as sc
from porter.exceptions import PorterException

import numpy as np

class FunctionService(BaseService):

    route_kwargs = {'methods': ['GET', 'POST'], 'strict_slashes': False}

    def __init__(self, action, function,
                 input_schema=None,
                 output_schema=None,
                 additional_checks=None,
                 **kwargs):
        self._action = action
        super().__init__(**kwargs)
        if not callable(function):
            raise ValueError('`function` must be callable')
        self.callable = function
        if input_schema is not None:
            self.add_request_schema('POST', input_schema)
        self.add_response_schema('GET', 200, sc.String())
        if output_schema is not None:
            self.add_response_schema('POST', 200, output_schema)
        if additional_checks is not None and not callable(additional_checks):
            raise ValueError('`additional_checks` must be callable')
        self.additional_checks = additional_checks

    @property
    def action(self):
        return self._action

    @property
    def status(self):
        return 'READY'

    def serve(self):
        if porter_api.request_method() == 'GET':
            return f"This endpoint is live. Send POST requests for '{self.action}'."
        data = self.get_post_data()
        if self.additional_checks is not None:
            self.additional_checks(data)
        out = self.callable(data)
        return out


def sum(x):
    return np.sum(x).tolist()

def prod(x):
    return np.prod(x).tolist()

def check_for_zeros(x):
    if 0 in x:
        raise PorterException('input cannot include zeros', code=422)

input_schema = sc.Array(item_type=sc.Number(), reference_name='InputSchema')
output_schema = sc.Number(reference_name='OutputSchema')
service_kw = dict(
    input_schema=input_schema,
    output_schema=output_schema,
    validate_request_data=True)

sum_service = FunctionService('sum', sum, name='math', api_version='v1', **service_kw)
prod_service = FunctionService('prod', prod, name='math', api_version='v1',
                               additional_checks=check_for_zeros, **service_kw)

app = ModelApp(
    [sum_service, prod_service],
    name='FunctionService Example',
    description='Expose arbitrary callable functions by subclassing BaseService.',
    expose_docs=True)

if __name__ == '__main__':
    app.run()
