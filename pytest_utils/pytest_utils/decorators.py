#Decorator for setting the max score of a test
from functools import wraps

def max_score(max_score):
    def wrapper(f):
        f.max_score = max_score
        return f
    return wrapper

#Optional decorator for setting the visibility of a test
#Options: 'visible', 'hidden', 'after_due_date', 'after_published'
def visibility(visibility):
    def wrapper(f):
        f.visibility = visibility
        return f
    return wrapper

#Optional decorator for adding extra tags to a test
#Should be an array of strings
def tags(tags):
    def wrapper(f):
        f.tags = tags
        return f
    return wrapper

def partial_score(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Execute the test function
        result = func(*args, **kwargs)
        # Assume the test function sets an attribute on itself for the partial score
        partial_score = getattr(func, 'partial_score', 0)
        explanation = getattr(func, 'explanation', 0)
        # Here, you could use the score for further logic or pass it to the grading mechanism
        return result
    return wrapper

def capture_assert_message(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except AssertionError as e:
            # Capture the assert message
            func.assert_message = str(e.args[0]) if e.args else ""
            # Optionally, you can re-raise the exception if you want the test to fail
            raise
    return wrapper


# comes from gradescope_utils
'''
class hide_errors(object):
    """Simple decorator to add a __hide_errors__ property to a function

    Usage: @hide_errors("Error message to be shown upon test failure")

    Used to hide the particular source of an error which caused a test to fail.
    Otherwise, a test's particular assertions can be seen by students.
    """

    def __init__(self, val="Test failed"):
        self.val = val

    def __call__(self, func):
        func.__hide_errors__ = self.val
        return func
'''


# Implementation as a function decorator
def hide_errors(message):
    def decorator(func):
        func.hide_errors = message
        return func
    return decorator

