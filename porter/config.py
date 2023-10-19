"""Configuration options."""

from functools import partial

from .utils import JSONProvider, AppEncoder

# JSON encoder used to serialize/deserialize request/response data.
# If used with porter's automatic validations this should return native-like
# types, such as dict, list, number, str, etc.
json_encoder = partial(JSONProvider, encoder=AppEncoder())

# Configurations for error responses.
# Including traceback and user data in responses is useful for debugging
# but not recommended for production apps
return_message_on_error = True
return_traceback_on_error = False
return_user_data_on_error = False

# Configurations for base response
return_request_id = True

# Support response compression
support_response_gzip = False
