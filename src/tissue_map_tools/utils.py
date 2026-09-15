import socket
import subprocess
def docstring_parameter(**kw):
    """Decorator to format docstrings with keyword arguments."""

    def decorator(func):
        if func.__doc__:
            func.__doc__ = func.__doc__.format(**kw)
        return func

    return decorator


def is_running_in_notebook() -> bool:
    """Returns True if running inside a Jupyter notebook/lab (or qtconsole),
    False for a plain script or a plain IPython terminal session."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        return shell.__class__.__name__ == "ZMQInteractiveShell"
    except ImportError:
        return False
    

def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0)) 
        return s.getsockname()[1]

