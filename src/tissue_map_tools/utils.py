def docstring_parameter(**kw):
    """Decorator to format docstrings with keyword arguments."""

    def decorator(func):
        if func.__doc__:
            func.__doc__ = func.__doc__.format(**kw)
        return func

    return decorator
