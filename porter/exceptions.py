class PorterException(Exception): 
    def __init__(self, *args, code):
        super().__init__(*args)
        self.code = code
