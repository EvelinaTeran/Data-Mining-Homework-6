import builtins

original_import = builtins.__import__  # Save the original import function

def restricted_import(*args, **kwargs):
    restricted_modules = kwargs.pop('restricted', [])  # Retrieve the list of restricted modules
    if args[0] in restricted_modules:
        raise ImportError(f"Import of {args[0]} is forbidden")
    return original_import(*args, **kwargs)


# ----------------------------------------------------------------------

import functools

def restrict_imports(restricted_list):
    def decorator(test_func):
        @functools.wraps(test_func)
        def wrapper(*args, **kwargs):
            # Override the built-in import with our custom function
            builtins.__import__ = functools.partial(restricted_import, restricted=restricted_list)
            try:
                result = test_func(*args, **kwargs)
            finally:
                # Restore the original import function after the test
                builtins.__import__ = original_import
            return result
        return wrapper
    return decorator

# ----------------------------------------------------------------------

import pytest

#@restrict_imports(['sklearn', 'numpy'])
@restrict_imports(['sklearn'])
def test_my_custom_algorithm():
    try:
        import sklearn
    except ImportError:
        assert False, "Import of 'sklearn' is restricted"

    """
    try:
        import numpy
        assert False, "Import should have been restricted"
    except ImportError:
        pass  # Expected to pass since numpy is restricted
    """

    import json  # This should work, json is not restricted
    assert True  # Indicate the test passed

"""
@restrict_imports(['pandas'])
def test_another_algorithm():
    try:
        import pandas
        assert False, "Import should have been restricted"
    except ImportError:
        pass  # Expected to pass since pandas is restricted

    import os  # This should work, os is not restricted
    assert True  # Indicate the test passed
"""
