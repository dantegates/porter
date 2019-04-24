from .utils import AppEncoder

json_encoder = AppEncoder

# Configurations for error responses.
# Including traceback and user data in responses is useful for debugging
# but not recommended for production apps
include_message = True
include_traceback = False
include_user_data = False
