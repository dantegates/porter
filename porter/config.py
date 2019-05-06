from .utils import AppEncoder

json_encoder = AppEncoder

# Configurations for error responses.
# Including traceback and user data in responses is useful for debugging
# but not recommended for production apps
return_message_on_error = True
return_traceback_on_error = False
return_user_data_on_error = False
return_request_id_on_error = False

# Configurations for prediction responses
return_request_id_with_prediction = False
