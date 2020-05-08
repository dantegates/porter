"""Generic HTTP exception for use with ``porter``."""

class PorterException(Exception): 
    """Generic HTTP Exception.

    Args:
        *args: Passed to ``Exception()``
        code (int): The HTTP status code.
    """
    def __init__(self, *args, code):
        super().__init__(*args)
        self.code = code
