class DocstringWriter:
    def __init__(self, doc):
        self.doc = doc

    def __call__(self, func):
        func.__doc__ = self.doc if self.doc else ""
        return func


def _param_freezer(*args, **kwargs):
    """Returns a decorator that freezes all parameters of the
    user function except the first one.
    """
    def decorating_function(func):

        def wrapper(var):
            return func(var, *args, **kwargs)
        return wrapper

    return decorating_function
