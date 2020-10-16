
Configuration
=============

porter responses can be configured using a global configuration under :mod:`porter.config`.


JSON Encoder
------------

``porter.config.json_encoder`` specifies how response data is converted to JSON.  The default encoder is :class:`porter.utils.AppEncoder`, which ensures that NumPy datatypes are converted to pure Python ones.


Error Responses
---------------

There are three settings that affect response data content for error conditions:

* ``porter.config.return_message_on_error`` (default: True): whether to include the error message.

* ``porter.config.return_traceback_on_error`` (default: False): whether to include the complete exception traceback.

* ``porter.config.return_user_data_on_error`` (default: False): whether to include the user, i.e. request, data.



Compression
-----------

porter supports gzip compression in request data by default.  Request data will be decompressed if the header ``Content-Encoding: gzip`` is included in the request.

Response data compression support is optional and disabled by default.  To enable this support, set ``porter.config.support_response_gzip = True``.  With this enabled, users can request compressed response data by including the header ``Accept-Encoding: gzip`` in the request.  If there is an error, the response will not be compressed.  Otherwise, the response will be compressed and the header ``Content-Encoding: gzip`` will be included in the response.
