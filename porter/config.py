from .utils import AppEncoder

json_encoder = AppEncoder

# Configurations for error responses.
# Including traceback and user data in responses is useful for debugging
# but not recommended for production apps
return_message_on_error = True
return_traceback_on_error = False
return_user_data_on_error = False

# Configurations for base response
return_request_id = True
