import os
import math
from typing import Type, Dict
import numpy as np
from numpy.typing import NDArray
import pytest
import json
import base64

import pickle

"""
def pytest_runtest_protocol(item, nextitem):
    reports = yield from pytest.runner.default_runtest_protocol(item, nextitem)
    if reports[1].failed:
        status = "failed"
    else:
        status = "passed"
    item.user_properties.append(("status", status))
    return reports

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    result = {'tests': []}
    for report in terminalreporter.stats.get('passed', []):
        result['tests'].append({
            'name': report.nodeid,
            'status': dict(report.user_properties).get('status', 'unknown'),
        })
    for report in terminalreporter.stats.get('failed', []):
        result['tests'].append({
            'name': report.nodeid,
            'status': dict(report.user_properties).get('status', 'unknown'),
        })
    with open('results.json', 'w') as f:
        json.dump(result, f, indent=4)
"""


"""
def dynamic_function_tracking(list_called_functions):
    # Use the fixture to list functions called in your_function
    called_functions_part1 = list_called_functions(Part1)
    print("==> Called functions in class Part1:", called_functions_part1)
    for func in called_functions_part1:
        func = track_calls(func)

    # Similarly for a class
    called_methods = list_called_functions(
        self.part1.train_simple_classifier_with_cv
    )
    print("Called methods in function :", called_methods)
"""


# ----------------------------------------------------------------------


@pytest.fixture
def capture_method_calls(mocker):
    def _capture(cls_to_wrap, module_path):
        # Create the wrapper and patch the class at the specified module path
        wrapper = MethodCallCapturingWrapper(cls_to_wrap)
        patched_class = mocker.patch(
            module_path + "." + cls_to_wrap.__name__, new=wrapper
        )

        # Return the wrapped instance for further use in tests
        return patched_class

    return _capture


def print_captured_calls(captured_methods):
    """
    Print details of captured method calls, with special handling for np.ndarray arguments.
    """
    for method_name, calls in captured_methods.items():
        # print(f"Method: {method_name}, Number of Calls: {len(calls)}")
        for i, call in enumerate(calls):
            args_str = ", ".join(_format_arg(arg) for arg in call.get("args", ()))
            kwargs_str = ", ".join(
                f"{k}={_format_arg(v)}" for k, v in call.get("kwargs", {}).items()
            )
            # print(f"  Call {i+1}: args: ({args_str}), kwargs: {{{kwargs_str}}}")
        # print()  # For better separation between methods


def _format_arg(arg):
    """
    Format an argument for printing. If it's an np.ndarray, print its shape.
    """
    if isinstance(arg, np.ndarray):
        return f"np.ndarray(shape={arg.shape})"
    elif isinstance(arg, list) and all(not isinstance(elem, list) for elem in arg):
        return f"1D list(length={len(arg)})"
    else:
        return str(arg)


def check_argument_usage(
    captured_methods, method_name, arg_name, expected_value=None, tolerance=None
):
    """
    Check if a specific argument was used in a method call and if it has a specific value.

    :param captured_methods: Dictionary of captured method calls.
    :param method_name: Name of the method to check.
    :param arg_name: Name of the argument to check.
    :param expected_value: (Optional) The expected value of the argument.
    :param tolerance: (Optional) The maximum absolute error if the argument is a float
    :return: Tuple (used, value_matched). 'used' is True if the argument was used,
             'value_matched' is True if the argument's value matches expected_value.
    """
    used = False
    value_matched = False

    for call in captured_methods.get(method_name, []):
        args, kwargs = call.get("args", ()), call.get("kwargs", {})

        # Check for the argument and compare values
        if isinstance(arg_name, int) and arg_name < len(args):
            used = True
            value_matched = _compare_values(args[arg_name], expected_value, tolerance)
        elif arg_name in kwargs:
            used = True
            value_matched = _compare_values(kwargs[arg_name], expected_value, tolerance)

        if used and (expected_value is None or value_matched):
            break

    return used, value_matched


def _compare_values(value, expected, tol):
    if tol is not None and all(isinstance(x, float) for x in [value, expected]):
        return math.isclose(value, expected, rel_tol=tol)
    elif (
        isinstance(value, list)
        and isinstance(expected, list)
        and len(value) == len(expected)
    ):
        return all(_compare_values(v, ev, tol) for v, ev in zip(value, expected))
    else:
        return value == expected


def assert_almost_equal(
    a: float, b: float, percent_tolerance: float, error_msg
) -> None:
    """
    Asserts that two floating point numbers are the same within a certain percentage.

    :param a: First floating point number.
    :param b: Second floating point number.
    :param percent_tolerance: Allowed percentage difference between the numbers.
    """
    tolerance = abs(a * percent_tolerance / 100)
    assert abs(a - b) <= tolerance, error_msg
    # assert abs(a - b) <= tolerance, f"Numbers {a} and {b} differ by more than {percent_tolerance}% tolerance"


# ----------------------------------------------------------------------



def save_dict(filenm, dct):
    with open(filenm, "wb") as file:
        pickle.dump(dct, file)


# Loading from a pickle file
def load_dict(filenm, dct):
    with open(filenm, "rb") as file:
        loaded_data = pickle.load(file)
    return loaded_data


def check_keys(correct_keys: list, student_keys: list, msg):
    # print(f"Check {msg}.keys()")
    for k in correct_keys:
        assert k in student_keys, f"Key '{k}' is not in student keys ({student_keys})"


def track_calls(func):
    call_log = []

    def wrapper(*args, **kwargs):
        call_log.append(func.__name__)
        return func(*args, **kwargs)

    wrapper.call_log = call_log
    return wrapper

def preprocess_yaml(input_file, output_file):
    with open(input_file, 'r') as file:
        data = yaml.safe_load(file)
    
    for question in data.get('questions', []):
        for part in question.get('parts', []):
            # Encode all answers
            #part['answer'] = encode_answer(part['answer'])
            part['answer'] = encode_data(part['answer'])
    
    with open(output_file, 'w') as file:
        yaml.dump(data, file, sort_keys=False)



# Example: Encoding with type hints
def encode_data(data):
    if isinstance(data, set):
        encoded = json.dumps({"type": "set", "data": list(data)})
    else:
        # print("data= ", data)
        encoded = json.dumps(data)  # Directly use JSON for other types
    return base64.b64encode(encoded.encode()).decode()

# Example: Decoding with type handling
def decode_data(encoded):
    decoded = json.loads(base64.b64decode(encoded).decode())
    # Not clear why the next two lines are needed
    if isinstance(decoded, dict) and decoded.get("type") == "set":
        return set(decoded["data"])
    if isinstance(decoded, str) and decoded[0:3] == 'set':
        return eval(decoded)
    return decoded



if __name__ == "__main__":
    starter_code()
