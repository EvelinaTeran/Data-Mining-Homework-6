import ast
import re
import inspect  # <<<<
import random
import math
import numpy as np
import yaml

# from pprint import pprint
# import matplotlib.pyplot as plt


def init_partial_score_dict() -> dict[str, float | int]:
    return {"nb_mismatches": 0, "nb_total": 0, "partial_score_frac": 0}


# ----------------------------------------------------------------------


def check_missing_keys(missing_keys, msg_list):
    if len(missing_keys) > 0:
        status = False
        msg_list.append(f"- Missing keys: {[repr(k) for k in missing_keys]}.")
    else:
        status = True
        msg_list.append("- No missing keys")
    return status, msg_list


# ----------------------------------------------------------------------
def check_float(i_el, s_el, rel_tol=1.0e-2, abs_tol=1.0e-5):
    status = True
    msg = ""

    if math.fabs(i_el) <= abs_tol:
        abs_err = math.fabs(i_el - s_el)
        status = True if abs_err < 1.0e-5 else False
    elif math.fabs((i_el - s_el) / i_el) < rel_tol:
        status = True
    else:
        status = False
        msg = f"Student element {s_el} has rel error > {100*rel_tol}% relative to instructor element {i_el}"
    return status, msg


# ----------------------------------------------------------------------


def check_int(i_el, s_el):
    status = True
    msg = ""
    if i_el != s_el:
        status = False
        msg = f"Student element {s_el} != instructor element {i_el}"
    return status, msg


# ----------------------------------------------------------------------


def check_list_float(i_arr, s_arr, rel_tol, abs_tol, ps_dict: dict[str, float | int]):
    """
    ps_dict : partial_score_dict
    """
    msg_list = []
    status = True
    ps_dict["nb_total"] += len(i_arr)

    for i_el, s_el in zip(i_arr, s_arr):
        status_, msg_ = check_float(i_el, s_el, rel_tol=rel_tol, abs_tol=abs_tol)
        if status_ is False:
            status = False
            msg_list.append(msg_)
            ps_dict["nb_mismatches"] += 1
    return status, "\n".join(msg_list)


# ----------------------------------------------------------------------


def check_list_int(i_arr, s_arr, ps_dict: dict[str, float | int]):
    """ """
    msg_list = []
    status = True
    ps_dict["nb_total"] += len(i_arr)

    for i_el, s_el in zip(i_arr, s_arr):
        status_, msg_ = check_int(i_el, s_el)
        if status_ is False:
            status = False
            msg_list.append(msg_)
            ps_dict["nb_mismatches"] += 1
    return status, "\n".join(msg_list)


# ----------------------------------------------------------------------


def check_set_int(i_set: set[int], s_set: set[int], ps_dict: dict[str, float | int]):
    """ """
    msg_list = []
    ps_dict["nb_total"] += len(i_set)

    # Elements in the instructor set not in the student set
    el_unique_to_i = i_set - s_set
    el_unique_to_s = s_set - i_set

    # Partial credits for sets
    # If the student has less elements than the teacher, subtrace the difference
    nb_deficit = len(el_unique_to_i)  # will be zero if the sets are equal

    # Number of student elements not in the instructor set are errors.
    nb_s_errors = len(el_unique_to_s)

    ps_dict["nb_mismatches"] += nb_deficit + nb_s_errors
    ps_dict["partial_score_frac"] = 1.0 - ps_dict["nb_mismatches"] / ps_dict["nb_total"]

    if nb_deficit + nb_s_errors > 0:
        status = False
        msg_list.append(f"Student has {nb_deficit} fewer elements than the instructor")
        msg_list.append(f"Student has {nb_s_errors} elements not in the instructor set")
    else:
        status = True
        msg_list.append("Student set is correct")

    return status, msg_list


# ----------------------------------------------------------------------
def check_str(i_str, s_str, str_choices: list[str] | None =None, remove_spaces: bool=None):
    status = True
    msg = ""
    str_choices = [clean_str_answer(s) for s in str_choices]
    i_str = clean_str_answer(i_str)
    s_str = clean_str_answer(s_str)

    #print("check_str: remove_spaces: ", remove_spaces)
    if remove_spaces is True:
        i_str = re.sub(r"\s+", "", i_str)
        s_str = re.sub(r"\s+", "", s_str)
        str_choices = [re.sub(r"\s+", "", el) for el in str_choices]

    if s_str in str_choices:
        #print(f"s_str: {s_str} is in {str_choices=}")
        s_str = i_str

    if i_str != s_str:
        status = False
        msg = f"String element mismatch. Instructor: {i_str}, Student: {s_str}"

    return status, msg


# ----------------------------------------------------------------------
def check_list_str(i_list, s_list, ps_dict: dict[str, float | int]):
    msg_list = []
    status = True
    ps_dict["nb_total"] += len(i_list)

    for i_el, s_el in zip(i_list, s_list):
        status_, msg_ = check_str(i_el, s_el)
        if status_ is False:
            status = False
            msg_list.append(msg_)
            ps_dict["nb_mismatches"] += 1
    return status, msg_list


# ----------------------------------------------------------------------
def check_dict_str_str(
    i_dict: dict[str, str], s_dict: dict[str, str], ps_dict: dict[str, float | int]
):
    """
    Note: dict[str, ???], ??? == str
    """
    status = True
    msg_list = []
    ps_dict["nb_total"] += len(i_dict)
    for k in i_dict.keys():
        status_, msg_ = check_str(i_dict[k], s_dict[k])
        if status_ is False:
            status = False
            msg_list.append(msg_)
            ps_dict["nb_mismatches"] += 1
    update_score(ps_dict)
    return status, msg_list


# ----------------------------------------------------------------------
def update_score(ps_dict: dict[str, float | int]) -> None:
    """ """
    ps_dict["partial_frac_score"] = 1.0 - ps_dict["nb_mismatches"] / ps_dict["nb_total"]


# ----------------------------------------------------------------------


def check_dict_str_float(
    keys, i_dict, s_dict, rel_tol, abs_tol, ps_dict: dict[str, float | int]
):
    msg_list = []
    status = True

    for k in keys:
        i_el = i_dict.get(k, None)
        s_el = s_dict.get(k, None)
        if i_el is None or s_el is None:
            continue
        status_, msg_ = check_float(i_el, s_el, rel_tol=rel_tol, abs_tol=abs_tol)
        if status_ is False:
            msg_list.append(msg_)
            status = False
            ps_dict["nb_mismatches"] += 1

    return status, msg_list


# ----------------------------------------------------------------------


def check_dict_int(keys, i_dict, s_dict, ps_dict: dict[str, float | int]):
    msg_list = []
    status = True

    for k in keys():
        i_el = i_dict.get(k, None)
        s_el = s_dict.get(k, None)
        if i_el is None or s_el is None:
            continue
        status_, msg_ = check_int(i_el, s_el)
        if status_ is False:
            msg_list.append(msg_)
            status = False
            ps_dict["nb_mismatches"] += 1

    return status, "\n".join(msg_list)


# ......................................................................


def is_explain(answer):
    """
    Is 'explain' in the cleaned answer string?
    Not currently used.
    """
    return "explain" in clean_str_answer(answer)


# ......................................................................


def clean_str_answer(answer):
    answer = answer.lower().strip()
    # Transform double spaces to single space
    answer = re.sub(r"\s+", " ", answer)
    return answer


# ......................................................................


def load_yaml_file(file_path):
    with open(file_path, "r") as file:
        questions_data = yaml.safe_load(file)
    return questions_data


def extract_config_dict():
    dct = {}
    config_dict_ = load_yaml_file("generator_config.yaml")
    test_structure = config_dict_.get("test_structure", None)
    dct["max_nb_words"] = (
        test_structure.get("max_nb_words", 10) if test_structure else 10
    )
    types = test_structure.get("types", {})
    eval_float = types.get("eval_float", {})
    dct["local_namespaces"] = eval_float.get("local_namespaces")
    return dct


config_dict = extract_config_dict()


# ----------------------------------------------------------------------
def fmt_ifstr(x):
    return repr(x) if isinstance(x, str) else str(x)


# ----------------------------------------------------------------------
def return_value(status, msg_list, s_answ, i_answ):
    """
    Used when checking the answer for a question,
    as opposed to the structure.
    """
    if status:
        msg_list.append("Answer is correct")
    else:
        msg_list.append("Answer is incorrect.")
    msg_list.append(f"Instructor answer: {fmt_ifstr(i_answ)}")
    msg_list.append(f"Student answer: {fmt_ifstr(s_answ)}")

    return status, "\n".join(msg_list)


# ----------------------------------------------------------------------
def are_sets_equal(set1, set2, rtol=1e-5, atol=1e-6):
    """
    Compares two sets of floats for equality within a relative and absolute tolerance.

    Args:
        set1: The first set of floats.
        set2: The second set of floats.
        rtol: The relative tolerance (default: 1e-5).
        atol: The absolute tolerance (default: 1e-8).

    Returns:
        True if the sets are equal within the specified tolerances, False otherwise.
    """
    if len(set1) != len(set2):
        return False  # Sets must have the same length
    for x, y in zip(set1, set2):
        if not np.isclose(x, y, rtol=rtol, atol=atol):
            return False
    return True


# ======================================================================
def check_answer_float(student_answer, instructor_answer, rel_tol, abs_tol):
    """
    Check answer correctness. Assume the structure is correct.
    """
    status, msg = check_float(
        instructor_answer, student_answer, rel_tol=rel_tol, abs_tol=abs_tol
    )
    return return_value(status, [msg], student_answer, instructor_answer)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_float(student_answer):
    if isinstance(student_answer, float):
        status = True
        msg_list = ["Answer is of type float as expected."]
    else:
        status = False
        msg_list = [
            f"Answer should be of type float. It is of type {repr(type(student_answer).__name__)}"
        ]
    return status, "\n".join(msg_list)


# ======================================================================
def check_answer_eval_float(
    student_answer, instructor_answer, local_vars_dict, rel_tol
):
    msg_list = []
    status = True
    s_answ = student_answer
    i_answ = instructor_answer
    # s_answ = s_answ.replace('^', '**')
    # s_answ = s_answ.replace('x', '*')
    # s_answ = s_answ.replace('X', '*')
    # i_answ = i_answ.replace('^', '**')
    random_values = {}
    local_dct = {}

    nb_evals = 3
    for _ in range(nb_evals):
        for var, (lower, upper) in local_vars_dict.items():
            random_values[var] = random.uniform(lower, upper)
            local_dct[var] = random_values[var]
        s_float = eval(s_answ, config_dict["local_namespaces"], local_dct)
        i_float = eval(i_answ, config_dict["local_namespaces"], local_dct)
        status, msg = check_float(i_float, s_float, rel_tol=rel_tol, abs_tol=1.0e-5)
        msg_list.append(msg)
        return return_value(status, msg_list, s_answ, i_answ)


# ======================================================================
def check_structure_eval_float(student_answer):
    if not isinstance(student_answer, str):
        return (
            False,
            "Student_answer is {student_answer}. Should be string defining a valid Python expression.",
        )
    try:
        ast.parse(student_answer, mode="eval")
        return True, "Valid python expression"
    except SyntaxError:
        print("===> expression is false")
        return False, "Your valid expression is not valid Python"


# ======================================================================


# def check_answer_dict_str_dict_str_list(student_answer, instructor_answer):
#     """
#     The type is a dict[str, dict[str, list]]

#     TODO: add exclusion keys for the outer dictionary
#     TODO: add exclusion keys for the inner dictionary. Establish notations

#     str_list of what? float? int? str? Or mixed types?
#     TODO: IDEA: for each element of the list, check the type and call the appropriate function
#     """
#     print(">>> assert_utilities ==> not handled")
#     return False, "Type 'dict_str_dict_str_list' NOT HANDLED!"

#     status = True
#     msg_list = []
#     ps_dict = init_partial_score_dict()

#     for k in instructor_answer.keys():
#         if not (isinstance(k, str) and isinstance(v, dict)):
#             return False

#         for inner_k, inner_v in v.items():
#             if not (isinstance(inner_k, str) and isinstance(inner_v, list)):
#                 return False

#     return True

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_dict_str_dict_str_list(student_answer, instructor_answer):
    """
    Check answer for type dict[str, dict[str, list]].
    Not handled because a list in itself should have a type
    Check against instructor keys.
    TODO: add inclusion and exclusion keys
    """
    msg_list = []
    status = True
    i_ans = instructor_answer
    s_ans = student_answer

    if not isinstance(student_answer, dict):
        return False, "Answer must a dict"

    missing_keys = set(i_ans.keys()) - set(s_ans.keys())
    if len(missing_keys) > 0:
        return False, f"- Missing keys: {[repr(k) for k in missing_keys]}."

    for k, v in instructor_answer.items():
        if not isinstance(v, dict):
            msg_list.append(f"- answer[{repr(k)}] must be of type 'dict'")
            status *= False
            continue
        # v is a dict
        for kk, vv in v.items():
            if not (isinstance(kk, str) and isinstance(vv, list)):
                msg_list.append(
                    f"- answer[{repr(k)}] must have keys of type 'str' and values of type 'list'"
                )
                status *= False

    if status is True:
        msg_list.append("Type 'dict[str, dict[str, list]' is correct.")

    return status, "\n".join(msg_list)


# ======================================================================
def check_answer_dict_str_dict_str_float(
    student_answer, instructor_answer, rel_tol, dict_float_choices, partial_score_frac: list[float]
):
    """
    The type is a dict[str, dict[str, list]]
    """
    status = True
    msg_list = []
    ps_dict = init_partial_score_dict()

    # Should go in structure check
    if not isinstance(student_answer, dict):
        return False

    for k, v in instructor_answer.items():
        # if not (isinstance(k, str) and isinstance(v, dict)):
        # return False

        # I should have a keys argument
        keys = instructor_answer.keys()
    
        s_answ = student_answer[k]
        if len(dict_float_choices) > 0 and k in dict_float_choices:
            for val in dict_float_choices[k]:
                status_, msg_list_ = check_dict_str_float(
                    keys, v, student_answer[k], rel_tol, 1.0e-5, ps_dict
                )
                if status_ is True:
                    msg_list_.append(f"Student answer {student_answer[k]} is within rel error of {rel_tol*100}%% of one of the accepted answers ({val})")
                    break
        else:
            status_, msg_list_ = check_dict_str_float(
                keys, v, student_answer[k], rel_tol, 1.0e-5, ps_dict
        )

        if status_ is False:
            status = status_
            msg_list.extend(msg_list_)

    partial_score_frac[0] = 1.0 - ps_dict["nb_mismatches"] / ps_dict["nb_total"]
    return return_value(status, msg_list, student_answer, instructor_answer)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_dict_str_dict_str_float(student_answer, instructor_answer):
    """
    Check the structure of the answer. Expected type: dict[str, dict[str, float]]
    """
    msg_list = []
    status = True
    i_ans = instructor_answer
    s_ans = student_answer

    # Assumes instructor answer is correct. Ideally, the keys should be listed
    # in the yaml file.
    missing_keys = set(i_ans.keys()) - set(s_ans.keys())
    if len(missing_keys) > 0:
        return False, f"- Missing keys: {[repr(k) for k in missing_keys]}."

    # print("answer keys: ", list(student_answer.keys()))
    # for k, v in instructor_answer.items():
    # print(f"key: {k}, value: {v}")

    for k, v in instructor_answer.items():
        if not isinstance(v, dict):
            msg_list.append(f"- answer[{repr(k)}] must be of type 'dict'")
            status *= False
            continue
        # v is a dict
        for kk, vv in v.items():
            if not (isinstance(kk, str) and isinstance(vv, float)):
                msg_list.append(
                    f"- answer[{repr(k)}] must have keys of type 'str' and values of type 'float'"
                )
                status *= False

    if status is True:
        msg_list.append("Type 'dict[str, dict[str, float]' is correct.")

    return status, "\n".join(msg_list)


# ======================================================================

'''
def check_answer_dict(student_answer, instructor_answer):
    # This function must be thoroughly debugged. I don't trust it.
    """
    Compares two generic dictionaries for equality down to two levels.
    ### NOT TESTED. Written by GPT-4, modified by GE

    Args:
        i_dict = dict1: The first dictionary.
        s_dict = dict2: The second dictionary.

    Returns:
        True if the dictionaries are equal down to two levels, False otherwise.

    Uses recursive calls. Can return status, but how to return msg_list?
    """
    msg_list = []
    s_dict = student_answer
    i_dict = instructor_answer
    print("Assert_utilitites: Type dict NOT HANDLED")
    return False, ""

    status = True
    msg_list = ""

    # Check the top-level keys match
    if set(i_dict.keys()) != set(s_dict.keys()):
        return False

    # Iterate through keys and compare values
    for key in i_dict:
        # If both values are dictionaries, compare their keys and values
        if isinstance(i_dict[key], dict) and isinstance(s_dict[key], dict):
            if not check_answer_dict(s_dict[key], i_dict[key]):
                return False
        else:
            # If the values are not both dictionaries, directly compare them
            if i_dict[key] != s_dict[key]:
                return False

    return return_value(status, msg_list, student_answer, instructor_answer)
'''


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
def check_key_structure(s_dict, i_dict):
    """
    Checks if the key structure of two dictionaries matches down to two levels,
    with the second dictionary (i_dict) considered as the gold standard.

    Args:
        s_dict: The first dictionary (student-generated).
        i_dict: The second dictionary (instructor-generated, gold standard).

    Returns:
        True if the key structures match down to two levels, False otherwise.
    """

    # Check the top-level keys match
    if set(s_dict.keys()) != set(i_dict.keys()):
        return False

    # Iterate through keys and check structures
    for key in i_dict:
        # If both values are dictionaries, compare their key sets
        if isinstance(s_dict.get(key), dict) and isinstance(i_dict[key], dict):
            if not check_key_structure(s_dict[key], i_dict[key]):
                return False
        elif isinstance(s_dict.get(key), dict) != isinstance(i_dict[key], dict):
            # One is a dict and the other is not, key structure does not match
            return False

    return True


def check_structure_dict(student_answer, instructor_answer):
    """
    Checks if the key structure of two dictionaries matches down to two levels,
    with the second dictionary (i_dict) considered as the gold standard.

    Args:
        s_dict: The first dictionary (student-generated).
        i_dict: The second dictionary (instructor-generated, gold standard).

    Returns:
        True if the key structures match down to two levels, False otherwise.
    """

    s_dict = s_answ = student_answer
    i_dict = i_answ = instructor_answer

    if not isinstance(s_answ, dict):
        return False, "- Answer must be of type 'dict'"

    # Check the top-level keys match
    missing_keys = set(s_answ.keys()) - set(i_answ.keys())
    if set(s_answ.keys()) != set(i_answ.keys()):
        if len(missing_keys) > 0:
            return False, f"- Missing keys: {[repr(k) for k in missing_keys]}."

    # Iterate through keys and check structures
    for key in i_answ:
        # If both values are dictionaries, compare their key sets
        if isinstance(s_answ.get(key), dict) and isinstance(i_answ[key], dict):
            # MISSING FUNCITON!
            ret = check_key_structure(s_dict[key], i_dict[key])
            if not ret:
                return ret
        elif isinstance(s_answ.get(key), dict) != isinstance(i_answ[key], dict):
            # One is a dict and the other is not, key structure does not match
            return (
                False,
                "- Mismatch! Student and instructor answers are of different types.",
            )

    return True, "The dictionary elements types matches that of the instructor"


# ======================================================================


def check_answer_str(student_answer, instructor_answer, str_choices: list[str], remove_spaces):
    """
    Arguments:
    - str_choices: check that the answer is one of str_choices if str_choices is not None
    """
    #print(f"check_answer_str, {remove_spaces=}")
    status, msg = check_str(instructor_answer, student_answer, str_choices, remove_spaces=remove_spaces)
    return return_value(status, [msg], student_answer, instructor_answer)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


# MUST FIX
def check_structure_str(student_answer, choices):
    """
    choices: list of strings
    """
    status = True
    msg_list = []
    choices = [clean_str_answer(c) for c in choices]

    # Ideally, should be done when yaml file is preprocessed
    # All strings should be lowered at that time.
    # choices = [clean_str_answer(s) for s in choices]

    if not isinstance(student_answer, str):
        status = False
        msg_list += ["- Answer must be of type 'str'"]
    else:
        msg_list += ["- type 'str' is correct"]
        # clean choices (lower, strip, '  ' -> ' ')
        student_answer = clean_str_answer(student_answer)

    if status and choices != []:
        if student_answer not in choices:
            status = False
            msg_list += [f"- Answer must be one of {choices}"]
        else:
            msg_list += [f"- Answer {repr(student_answer)} is among the valid choices"]

    # print("\n".join(msg_list))
    return status, "\n".join(msg_list)


# ======================================================================


def check_answer_explain_str(student_answer, instructor_answer):
    msg_list = []
    status = True
    return return_value(status, msg_list, student_answer, instructor_answer)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_explain_str(student_answer):
    """
    The type is an explain_str
    The string should have a minimum number of words stored in "type_handlers.yaml"
    """
    msg_list = []
    status = True
    if isinstance(student_answer, str):
        status = True
        msg_list.append("- Type 'str' is correct")
    else:
        status = False
        msg_list.append("- Type must be of type 'str'")

    # Check the number of words in the string
    max_nb_words = config_dict["max_nb_words"]

    if status:
        is_nb_words = len(student_answer.split()) >= max_nb_words
        if is_nb_words:
            msg_list.append("- Length of answer is sufficient")
        else:
            status = False
            msg_list.append(f"- Length of answer must be > {max_nb_words}")

    return status, "\n".join(msg_list)


# ======================================================================

# IMPLEMENT partial_scoring. Add option in yaml file to turn it off. On by default .
# NOT DONE


def check_answer_set_str(
    student_answer, instructor_answer, partial_score_frac: list[float], choices=None
):
    """
    s_answ: student answer: set of strings
    i_answ: instructor answer: set of strings
    choices: one of several choices
    """
    msg_list = []
    status = True
    ps_dict = init_partial_score_dict()
    if choices is None:
        choices = []

    s_answ = {i.lower().strip() for i in student_answer}
    i_answ = {i.lower().strip() for i in instructor_answer}

    # TODO: How to compare two sets with strings for equality?

    # Only consider elements in choices
    # Each choice[i] is a list (of alternatives?)
    if choices and isinstance(choices[0], list):
        choices = [set(c) for c in choices]

    # TODO: do not consider elements in `exclude_list`
    # TODO: only consider elements in `include_list`
    # TODO: I should use **kwargs to simplify code

    # print("===> set_str, after set, choices: ", choices)

    if choices != [] and isinstance(choices[0], set):
        for i, a_set in enumerate(choices):
            choices[i] = {clean_str_answer(el) for el in a_set}

    if choices and isinstance(choices[0], set):
        status = True if s_answ in choices else False
    else:
        status = s_answ == i_answ

    msg_list.append("Strings are lower-cased and stripped")
    if choices and isinstance(choices[0], set):
        msg_list.append(f"Student answer is one of {choices}")
    return return_value(status, msg_list, s_answ, i_answ)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_set_str(student_answer):
    """ """
    msg_list = []
    status = True

    if isinstance(student_answer, set) or isinstance(student_answer, list):
        status = True
        msg_list.append("- Type is either 'list' or 'set' (correct).")
    else:
        status = False
        msg_list.append("- Answer must be of type 'set' or 'list'.")

    if status:
        are_all_str = True
        for s in student_answer:
            if not isinstance(s, str):
                msg_list.append("- Set element {repr(s)} must be of type 'str'")
                status = False
                are_all_str = False

    if are_all_str:
        msg_list.append("- All elements are 'str', as required")
        status = True

    return status, "\n".join(msg_list)


# ======================================================================


def check_answer_dict_str_set(student_answer, instructor_answer):
    """
    student answer: dictionary with keys:str, values: a set of objects
    instructor answer: dictionary with keys:str, values: a set of objects

    Even if the student returns a dict[str] = list, the list is cast to a set
    """
    # print("AssertUtilities: type dict_str_set NOT HANDLED")
    return False, ""

    # msg_list = []
    # status = True
    # for k in instructor_answer.keys():
    #     s_val = student_answer[k]
    #     i_val = instructor_answer[k]
    #     status *= set(s_val) == set(i_val)
    #  return return_value(status, msg_list, student_answer, instructor_answer


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_dict_str_set(student_answer, instructor_answer):
    """
    TODO: provide a list of keys to check as an argument keys (default None)
    """
    status = True
    msg_list = []

    if not isinstance(student_answer, dict):
        msg_list.append("- Answer should be of type 'dict'.")
    else:
        msg_list.append("- Type 'dict' is correct")

    keys = set(instructor_answer.keys())
    student_keys = set(student_answer.keys())
    missing_keys = keys - student_keys

    if status:
        if len(missing_keys) > 0:
            msg_list.append(f"- Missing keys: {[repr(k) for k in missing_keys]}.")
            status = False
        else:
            msg_list.append("- No missing keys")

    if status:
        is_item_type_list = True
        for k, v in student_answer.items():
            if k in keys:
                if not isinstance(v, (set, list)):
                    msg_list.append(
                        f"- Answer[{repr(k)}] must be of type 'set' or 'list'."
                    )
                    # The answer is cast to a set when checked for accuracy
                    status = False
                    is_item_type_list = False

        if is_item_type_list:
            msg_list.append(
                "- All list elements are of type 'list' or 'set' as expected"
            )
            status = True

    return status, "\n".join(msg_list)


# ======================================================================


def check_answer_dict_str_set_int(student_answer, instructor_answer, keys=None):
    """ """
    msg_list = []
    status = True
    ps_dict = init_partial_score_dict()
    status, msg_list = check_set_int(
        set(student_answer), set(instructor_answer), ps_dict
    )
    return return_value(status, msg_list, student_answer, instructor_answer)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_dict_str_set_int(student_answer, instructor_answer, keys=None):
    """
    TODO: provide a list of keys to check as an argument keys (default None)
    Check that the outer dict keys are correct
    keys: only check the keys in the keys argument
    """
    status = True
    msg_list = []

    if not isinstance(student_answer, dict):
        msg_list.append("Answer should be of type 'dict'.")
    else:
        msg_list.append("Type 'dict' is correct.")

    # Not clear that instructor should be an argument to the function
    keys = set(instructor_answer.keys()) if keys is None else keys

    student_keys = set(student_answer.keys())
    missing_keys = keys - student_keys
    status, msg_list = check_missing_keys(missing_keys, msg_list)

    if status:
        for k in keys:
            if isinstance(student_answer[k], int):
                msg_list.append("All set elements must be type 'int'.")

    return status, "\n".join(msg_list)


# ======================================================================


def check_answer_dict_str_float(
    student_answer, instructor_answer, rel_tol, keys, dict_float_choices, partial_score_frac: list[float]
):
    """
    student answer: dictionary with keys:str, values: an ndarray
    instructor answer: dictionary with keys:str, values: a set of objects
    rel_tol: tolerance on the matrix norm
    keys: None if all keys should be considered
    """
    # print("==> check_answer_dict_str_float")
    msg_list = []
    status = True
    keys = list(instructor_answer.keys()) if keys is None else keys
    ps_dict = init_partial_score_dict()
    ps_dict["nb_total"] = len(keys)

    # Need an exception in case the student key is not found
    for k in keys:
        s_float = student_answer[k]
        i_float = instructor_answer[k]

        if len(dict_float_choices) > 0 and k in dict_float_choices:
            for val in dict_float_choices[k]:
                if val == 'i': # use instructor answer
                    val = i_float
                status_, msg_list_ = check_float(
                    s_float, val, rel_tol, 1.0e-5
                )
                if status_ is True:
                    break
        else:
            status_, msg_ = check_float(i_float, s_float, rel_tol=rel_tol, abs_tol=1.e-6)

        if status_ is False:
            status = False
            ps_dict["nb_mismatches"] += 1
            msg_list.append(msg_)

    partial_score_frac[0] = 1.0 - ps_dict["nb_mismatches"] / ps_dict["nb_total"]
    return return_value(status, msg_list, student_answer, instructor_answer)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_dict_str_float(student_answer, instructor_answer, keys=None):
    """
    student answer: dictionary with keys:str, values: float
    instructor answer: dictionary with keys:str, values: a set of objects
    rel_tol: tolerance on the matrix norm
    keys: None if all keys should be considered
    """
    status = True
    msg_list = []

    if status and not isinstance(student_answer, dict):
        msg_list += ["Student answer should be a dict"]
        status = False

    if status:
        keys = keys if keys else list(instructor_answer.keys())
        instructor_keys = set(keys)
        instructor_answer = {k: v for k, v in instructor_answer.items() if k in keys}
        student_keys = set(student_answer.keys())
        missing_keys = list(instructor_keys - student_keys)

        if len(missing_keys) > 0:
            msg_list.append(f"- Missing keys: {[repr(k) for k in missing_keys]}.")
            status = False
        else:
            msg_list.append("- No missing keys.")

    if status:
        # some keys are filtered. Student is allowed to have
        # keys not in the instructor set
        for k, v in instructor_answer.items():
            vs = student_answer[k]
            if not isinstance(vs, float):
                msg_list.append(f"- answer[{repr(k)}] should be a float.")
                status = False

        if status:
            msg_list.append("- All elements are of type float as expected.")

    if status:
        msg_list.append("Type 'dict[str, float]' is correct")

    return status, "\n".join(msg_list)


# ======================================================================


def check_answer_dict_str_ndarray(
    student_answer, instructor_answer, rel_tol, keys, partial_score_frac: list[float]
):
    """
    student answer: dictionary with keys:str, values: an ndarray
    instructor answer: dictionary with keys:str, values: a set of objects
    rel_tol: tolerance on the matrix norm
    keys: None if all keys should be considered
    """
    msg_list = []
    status = True
    msg_list.append("Check array norms")

    keys = list(instructor_answer.keys()) if keys is None else keys
    ps_dict = init_partial_score_dict()
    ps_dict["nb_total"] = len(keys)

    # Need an exception in case the student key is not found
    i_norms = {k: np.linalg.norm(instructor_answer[k]) for k in keys}
    s_norms = {k: np.linalg.norm(student_answer[k]) for k in keys}

    status, msg_list = check_dict_str_float(
        keys, i_norms, s_norms, rel_tol, 1.0e-5, ps_dict
    )
    partial_score_frac[0] = 1.0 - ps_dict["nb_mismatches"] / ps_dict["nb_total"]
    return return_value(status, msg_list, s_norms, i_norms)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_dict_str_ndarray(student_answer, instructor_answer, keys=None):
    """
    student answer: dictionary with keys:str, values: an ndarray
    instructor answer: dictionary with keys:str, values: a set of objects
    keys: None if all keys should be considered
    """
    status = True
    msg_list = []

    if not isinstance(instructor_answer, dict):
        msg_list += ["Instructor answer should be a dict"]
        status = False

    if status and not isinstance(student_answer, dict):
        msg_list += ["Student answer should be a dict"]
        status = False

    if status:
        keys = keys if keys else list(instructor_answer.keys())
        instructor_keys = set(keys)
        instructor_answer = {k: v for k, v in instructor_answer.items() if k in keys}
        student_keys = set(student_answer.keys())
        missing_keys = list(instructor_keys - student_keys)

        if len(missing_keys) > 0:
            msg_list.append(f"- Missing keys: {[repr(k) for k in missing_keys]}.")
            status = False
        else:
            msg_list.append("- No missing keys.")

    if status:
        # some keys are filtered. Student is allowed to have
        # keys not in the instructor set
        for k, v in instructor_answer.items():
            vs = student_answer[k]
            if not isinstance(vs, type(np.zeros(1))):
                msg_list.append(f"- answer[{repr(k)}] should be a numpy array.")
                status = False

        if status:
            msg_list.append("- All elements are of type ndarray as expected.")

    if status:
        msg_list.append("Type 'dict[str, ndarray]' is correct")

    return status, "\n".join(msg_list)


# ======================================================================
def check_answer_dict_tuple_int_ndarray(
    student_answer, instructor_answer, rel_tol, keys, partial_score_frac: list[float]
):
    """
    Similar to check_answer_dict_str_ndarray
    student answer: dictionary with keys:str, values: an ndarray
    instructor answer: dictionary with keys:str, values: a set of objects
    rel_tol: tolerance on the matrix norm
    keys: None if all keys should be considered
    partial_score_frac: [float]
    """

    msg_list = ["Check array norms"]
    status = True
    i_dict_norm = {}
    s_dict_norm = {}
    keys = list(instructor_answer.keys()) if keys is None else keys
    sub_instructor_answer = {k: instructor_answer[k] for k in keys}

    # print("Assert_utilities, type dict_tuple_int_ndarray NOT HANDLED")
    # return False, ""

    ps_dict = init_partial_score_dict()
    ps_dict["total_nb"] = len(sub_instructor_answer)

    # Need an exception in case the student key is not found
    for k in sub_instructor_answer.keys():
        if k not in student_answer:
            status = False
            msg_list.append(f"The key {k} is missing")
            break
        s_arr = student_answer[k]
        i_arr = instructor_answer[k]
        if s_arr.shape != i_arr.shape:
            status = False
            msg_list.append(
                f"key: {k}, incorrect shape {s_arr.shape}, should be {i_arr.shape}."
            )
        i_dict_norm[k] = s_norm = np.linalg.norm(s_arr)
        s_dict_norm[k] = i_norm = np.linalg.norm(i_arr)
        status_, msg = check_float(i_norm, s_norm, rel_tol, abs_tol=1.0e-5)
        if status_ is False:
            status = False
            msg_list.append(msg)

    partial_score_frac[0] = 1.0 - ps_dict["nb_mismatches"] / ps_dict["nb_total"]

    return return_value(status, msg_list, student_answer, instructor_answer)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_dict_tuple_int_ndarray(student_answer, instructor_answer, keys):
    """ """
    status = True
    msg_list = []

    if not isinstance(instructor_answer, dict):
        msg_list += ["Instructor answer should be a dict"]
        status = False

    if status and not isinstance(student_answer, dict):
        msg_list += ["Student answer should be a dict"]
        status = False

    for key in student_answer.keys():
        if not isinstance(key, tuple):
            status = False
            msg_list += [
                f"key {key} should be of type 'tuple', but is type {type(key).__name__}."
            ]

    if status:
        keys = keys if keys else list(instructor_answer.keys())
        instructor_keys = set(keys)
        instructor_answer = {k: v for k, v in instructor_answer.items() if k in keys}
        student_keys = set(student_answer.keys())
        missing_keys = list(instructor_keys - student_keys)

        if len(missing_keys) > 0:
            msg_list.append(f"- Missing keys: {[repr(k) for k in missing_keys]}.")
            status = False
        else:
            msg_list.append("- No missing keys.")

    if status:
        # some keys are filtered. Student is allowed to have
        # keys not in the instructor set
        for k, v in student_answer.items():
            if not isinstance(k, tuple):
                status = False
                msg_list.append("At least one of your keys is not of type 'tuple'.")
                break
            for el in k:
                if not isinstance(el, int):
                    status = False
                    msg_list.append("At least one element of one key is not an 'int'.")
                    break
            vs = student_answer[k]
            if not isinstance(vs, type(np.zeros(1))):
                msg_list.append(f"- answer[{repr(k)}] should be a numpy array.")
                status = False

        if status:
            msg_list.append(
                "- All keys are tuples of ints and values are of type ndarray as expected."
            )

    if status:
        msg_list.append("Type 'dict[tuple[int], ndarray]' is correct")

    return status, "\n".join(msg_list)


# ----------------------------------------------------------------------


def check_answer_dict_int_ndarray(student_answer, instructor_answer, rel_tol, keys):
    """
    Similar to check_answer_dict_str_ndarray
    student answer: dictionary with keys:str, values: an ndarray
    instructor answer: dictionary with keys:str, values: a set of objects
    rel_tol: tolerance on the matrix norm
    keys: None if all keys should be considered
    """
    msg_list = []
    status = True
    i_dict_norm = {}
    s_dict_norm = {}
    keys = list(instructor_answer.keys()) if keys is None else keys

    # print("Assert_utilities, type dict_int_ndarray NOT HANDLED")
    # return False, ""

    msg_list.append(f"We are comparing array norms with a {rel_tol} relative accuracy.")

    # Need an exception in case the student key is not found
    for k in keys:
        s_arr = student_answer[k]
        i_arr = instructor_answer[k]
        if s_arr.shape != i_arr.shape:
            status = False
            msg_list.append(
                f"key: {k}, incorrect shape {s_arr.shape}, should be {i_arr.shape}."
            )
            continue
        s_norm = np.linalg.norm(s_arr)
        i_norm = np.linalg.norm(i_arr)
        i_dict_norm[k] = i_norm
        s_dict_norm[k] = s_norm
        if i_norm < 1.0e-5:
            abs_err = math.fabs(s_norm - i_norm)
            if abs_err > 1.0e-5:
                status = False
                msg_list.append(f"- key {k} has a norm with absolute error > 1.e-5")
        else:
            rel_err = math.fabs(s_norm - i_norm) / math.fabs(i_norm)
            if rel_err > rel_tol:
                status = False
                msg_list.append(
                    f"key: {k}, L2 norm is not within {int(100*rel_tol)}%\n\
                                relative error of the correct norm of {i_norm}."
                )

    return return_value(status, msg_list, student_answer, instructor_answer)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_dict_int_ndarray(student_answer, instructor_answer, keys=None):
    """
    student answer: dictionary with keys:str, values: an ndarray
    instructor answer: dictionary with keys:str, values: a set of objects
    rel_tol: tolerance on the matrix norm
    keys: None if all keys in the instructor_answer should be considered (list of ints)
    """
    status = True
    msg_list = []

    if not isinstance(instructor_answer, dict):
        msg_list += ["Instructor answer should be a dict"]
        status = False

    if status and not isinstance(student_answer, dict):
        msg_list += ["Student answer should be a dict"]
        status = False

    for key in student_answer.keys():
        if not isinstance(key, int):
            status = False
            msg_list += [
                f"key {key} should be of type 'int', but is type {type(key).__name__}."
            ]

    if status:
        keys = keys if keys else list(instructor_answer.keys())
        instructor_keys = set(keys)
        instructor_answer = {k: v for k, v in instructor_answer.items() if k in keys}
        student_keys = set(student_answer.keys())
        missing_keys = list(instructor_keys - student_keys)

        if len(missing_keys) > 0:
            msg_list.append(f"- Missing keys: {[repr(k) for k in missing_keys]}.")
            status = False
        else:
            msg_list.append("- No missing keys.")

    if status:
        # some keys are filtered. Student is allowed to have
        # keys not in the instructor set
        for k, v in instructor_answer.items():
            vs = student_answer[k]
            if not isinstance(vs, type(np.zeros(1))):
                msg_list.append(f"- answer[{repr(k)}] should be a numpy array.")
                status = False

        if status:
            msg_list.append("- All elements are of type ndarray as expected.")

    if status:
        msg_list.append("Type 'dict[str, ndarray]' is correct")

    return status, "\n".join(msg_list)


# ======================================================================
def check_answer_dict_int_list(student_answer, instructor_answer, keys):
    """
    Similar to check_answer_dict_str_ndarray
    list of floats (if not specified)
    student answer: dictionary with keys:str, values: an ndarray
    instructor answer: dictionary with keys:str, values: a set of objects
    keys: None if all keys should be considered

    # HOW TO CHECK?
    """
    msg_list = []
    status = True
    keys = list(instructor_answer.keys()) if keys is None else keys

    # print("Assert_utilities, type dict_int_list NOT HANDLED")
    return False, ""

    # Need an exception in case the student key is not found
    for k in keys:
        s_arr = student_answer[k]
        i_arr = instructor_answer[k]
        if s_arr.shape != i_arr.shape:
            status = False
            msg_list.append(
                f"key: {k}, incorrect shape {s_arr.shape}, should be {i_arr.shape}."
            )
        for i_el, s_el in zip(i_arr, s_arr):
            if i_el == s_el:
                status = True
            else:
                status = False
                msg_list.append(
                    f"Elements not equal (instructor/student): {i_el}/{s_el}"
                )
        # s_norm = np.linalg.norm(s_arr)
        # i_norm = np.linalg.norm(i_arr)
        # i_dict_norm[k] = i_norm
        # s_dict_norm[k] = s_norm
        # rel_err = math.fabs(s_norm - i_norm) / math.fabs(i_norm)
        # if rel_err > rel_tol:
        # status = False
        # msg_list.append(f"key: {key}, L2 norm is not within {int(100*rel_tol)}%\n\
        # relative error of the correct norm of {i_norm}.")

    return return_value(status, msg_list, student_answer, instructor_answer)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
def check_structure_dict_int_list(student_answer, instructor_answer, keys=None):
    """
    student answer: dictionary with keys:str, values: an ndarray
    list of floats (if not specified)
    instructor answer: dictionary with keys:str, values: a set of objects
    keys: None if all keys in the instructor_answer should be considered (list of ints)
    """
    status = True
    msg_list = []

    if not isinstance(instructor_answer, dict):
        msg_list += ["Instructor answer should be a dict"]
        status = False

    if status and not isinstance(student_answer, dict):
        msg_list += ["Student answer should be a dict"]
        status = False

    # I am not handling the keys argument yet <<<<<<
    # Check the length of the lists (NOT DONE) <<<<<
    # I could convert list to ndarray and call the function with NDARRAY for checking.
    # If the list cannot be converted, it has the wrong format. So use an try/except.

    if status:
        # some keys are filtered. Student is allowed to have
        # keys not in the instructor set
        for k in instructor_answer.keys():
            key = student_answer.get(k, None)
            if key is None:
                status = False
                msg_list.append(f"Key {k} is missing from student answer")
                continue
            vs = student_answer[k]
            if not isinstance(vs, list):
                status = False
                msg_list.append(
                    f"student_answer[{k}] is not type 'list'. Cannot proceed with answer check."
                )
            for el in vs:
                if not isinstance(el, float):
                    status = False
                    msg_list.append(
                        f"student_answer[{k}] is a list with at least one non-float element. Cannot proceed with answer check."
                    )
                    break

        if status:
            msg_list.append("- All elements are of type list of float as expected.")

    if status:
        msg_list.append("Type 'dict[str, list]' is correct")

    return status, "\n".join(msg_list)


# ======================================================================
def check_answer_dict_int_list_float(
    student_answer, instructor_answer, keys, rel_tol, partial_score_frac: list[float]
):
    """
    Similar to check_answer_dict_str_ndarray
    list of floats (if not specified)
    student answer: dictionary with keys:str, values: an ndarray
    instructor answer: dictionary with keys:str, values: a set of objects
    keys: None if all keys should be considered

    # HOW TO CHECK?
    """
    msg_list = []
    status = True
    keys = list(instructor_answer.keys()) if keys is None else keys
    ps_dict = init_partial_score_dict()
    ps_dict["nb_total"] = len(keys)

    # Need an exception in case the student key is not found
    for k in keys:
        s_arr = student_answer[k]
        i_arr = instructor_answer[k]
        status_, msg_list_ = check_list_float(
            i_arr, s_arr, rel_tol=rel_tol, abs_tol=1.0e-6, ps_dict=ps_dict
        )
        if status_ is False:
            status = False
            msg_list.extend(msg_list_)

    partial_score_frac[0] = 1.0 - ps_dict["nb_mismatches"] / ps_dict["nb_total"]
    return return_value(status, msg_list, student_answer, instructor_answer)


# ======================================================================


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_dict_int_list_float(student_answer, instructor_answer, keys=None):
    """
    student answer: dictionary with keys:str, values: an ndarray
    list of floats (if not specified)
    instructor answer: dictionary with keys:str, values: a set of objects
    keys: None if all keys in the instructor_answer should be considered (list of ints)
    """
    status = True
    msg_list = []

    if not isinstance(instructor_answer, dict):
        msg_list += ["Instructor answer should be a dict"]
        status = False

    if status and not isinstance(student_answer, dict):
        msg_list += ["Student answer should be a dict"]
        status = False

    keys = list(instructor_answer.keys()) if keys is None else keys
    sub_instructor_answer = {k: instructor_answer[k] for k in keys}

    # I am not handling the keys argument yet <<<<<<
    # Check the length of the lists (NOT DONE) <<<<<
    # I could convert list to ndarray and call the function with NDARRAY for checking.
    # If the list cannot be converted, it has the wrong format. So use an try/except.

    if status:
        # some keys are filtered. Student is allowed to have
        # keys not in the instructor set
        for k in sub_instructor_answer.keys():
            key = student_answer.get(k, None)
            if key is None:
                status = False
                msg_list.append(f"Key {k} is missing from student answer.")
                continue
            vs = student_answer[k]
            if not isinstance(vs, list):
                status = False
                msg_list.append(
                    f"student_answer[{k}] is not type 'list'. Cannot proceed with answer check."
                )
            for el in vs:
                if not isinstance(el, float):
                    status = False
                    msg_list.append(
                        f"student_answer[{k}] is a list with at least one non-float element. Cannot proceed with answer check."
                    )
                    break

        if status:
            msg_list.append("- All elements are of type list[float] as expected.")

    if status:
        msg_list.append("Type 'dict[str, list[float]]' is correct.")

    return status, "\n".join(msg_list)


# ======================================================================


def check_answer_list_int(
    student_answer, instructor_answer, partial_score_frac: list[float]
):
    """
    Check that all elements in the list have matching norms
    """
    msg_list = []
    status = True
    answ_eq_len = len(student_answer) == len(instructor_answer)
    ps_dict = init_partial_score_dict()
    ps_dict["nb_total"] = len(instructor_answer)

    if answ_eq_len:
        status, msg_list_ = check_list_int(student_answer, instructor_answer, ps_dict)
        msg_list.extend(msg_list_)

    if not status:
        msg_list.append("Some elements are incorrect")

    partial_score_frac[0] = 1.0 - ps_dict["nb_mismatch"] / ps_dict["nb_total"]
    return return_value(status, msg_list, student_answer, instructor_answer)




# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_list_int(student_answer, instructor_answer):
    """
    Check that elements in the list are ndarrays
    """
    status = True
    msg_list = []

    if not isinstance(student_answer, list):
        status = False
        msg_list.append(
            f"- The answer should be of type 'list'; your type is {type(student_answer).__name__}"
        )
    else:
        msg_list.append("- The answer is type list. Correct.")

    # Check length of list
    if status:
        if len(student_answer) != len(instructor_answer):
            status = False
            msg_list.append(
                [
                    f"- The length of your list is incorrect. Your list length is {len(student_answer)}.",
                    "The length should be {len(instructor_answer)}.",
                ]
            )
        else:
            msg_list.append("- The length of the list is correct.")

    if status:
        for s_arr in instructor_answer:
            if not isinstance(s_arr, int):
                status = False
                msg_list.append("- Element {i} of your list should be of type 'int'.")

    if status:
        msg_list.append("- All list elements are type 'int'.")

    return status, "\n".join(msg_list)

# ======================================================================
def check_answer_list_float(
        student_answer, instructor_answer, rel_tol, monotone_increasing=None, ps_dict: list[float|None]=None
):
    """
    Check that all elements in the list have matching norms
    Arguments: 
    - monotone_increasing: True/False. Default: None (ignore it). 
    """
    msg_list = []
    status = True
    answ_eq_len = len(student_answer) == len(instructor_answer) # checked in structure
    ps_dict = init_partial_score_dict()
    ps_dict["nb_total"] = len(instructor_answer)

    if answ_eq_len and (monotone_increasing is None or monotone_increasing is False):
        status, msg_list_ = check_list_float(student_answer, instructor_answer, rel_tol=rel_tol, abs_tol=1.e-6, ps_dict=ps_dict)
        msg_list.extend(msg_list_)
    elif monotone_increasing is True:
        # Check whether the list is monotone incrreasing. If not, fail. 
        val = student_answer[0]
        for el_val in student_answer[1:]:
            if el_val >= val:
                continue
            else:
                status = False
                msg_list.append("The answer is not monotonically increasing")

    if not status:
        msg_list.append("Some elements are incorrect")

    if monotone_increasing: 
        partial_score_frac[0] = 1.0
    else:
        partial_score_frac[0] = 1.0 - ps_dict["nb_mismatch"] / ps_dict["nb_total"]

    return return_value(status, msg_list, student_answer, instructor_answer)

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

def check_structure_list_float(student_answer, instructor_answer):
    """
    Check that elements in the list are ndarrays
    """
    status = True
    msg_list = []

    if not isinstance(student_answer, list):
        status = False
        msg_list.append(
            f"- The answer should be of type 'list'; your type is {type(student_answer).__name__}"
        )
    else:
        msg_list.append("- The answer is type list. Correct.")

    # Check length of list
    if status:
        if len(student_answer) != len(instructor_answer):
            status = False
            msg_list.append(
                [
                    f"- The length of your list is incorrect. Your list length is {len(student_answer)}.",
                    "The length should be {len(instructor_answer)}.",
                ]
            )
        else:
            msg_list.append("- The length of the list is correct.")

    if status:
        for s_arr in instructor_answer:
            if not isinstance(s_arr, float):
                status = False
                msg_list.append("- Element {i} of your list should be of type 'float'.")

    if status:
        msg_list.append("- All list elements are type 'float'.")

    return status, "\n".join(msg_list)

# ======================================================================


def check_answer_list_ndarray(
    student_answer, instructor_answer, rel_tol, partial_score_frac: list[float]
):
    """
    rel_tol: max relative error on the L2 norm
    Check that all elements in the list have matching norms
    """
    msg_list = []
    status = True
    answ_eq_len = len(student_answer) == len(instructor_answer)
    i_norm_list = []
    s_norm_list = []
    ps_dict = init_partial_score_dict()
    ps_dict["nb_total"] = len(instructor_answer)

    if answ_eq_len:
        for s_arr, i_arr in zip(student_answer, instructor_answer):
            s_norm = np.linalg.norm(s_arr)
            i_norm = np.linalg.norm(i_arr)
            i_norm_list.append(i_norm)
            s_norm_list.append(s_norm)
            #print(
            #   "IMPROVE: could first create a list of norms, and call check_list_float"
            #)
            status_, msg = check_float(i_norm, s_norm, rel_tol, abs_tol=1.0e-5)
            if status_ is False:
                status = False
                msg_list.append([msg])
                ps_dict["nb_mismatches"] += 1

    if not status:
        msg_list.append("Replace the arrays by their norms")

    partial_score_frac[0] = 1.0 - ps_dict["nb_mismatches"] / ps_dict["nb_total"]
    return return_value(status, msg_list, s_norm_list, i_norm_list)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_list_ndarray(student_answer, instructor_answer):
    """
    Check that elements in the list are ndarrays
    """
    status = True
    msg_list = []

    if not isinstance(student_answer, list):
        status = False
        msg_list.append(
            f"- The answer should be of type 'list'; your type is {type(student_answer)}"
        )
    else:
        msg_list.append("- The answer is type list. Correct.")

    # Check length of list
    if status:
        if len(student_answer) != len(instructor_answer):
            status = False
            msg_list.extend(
                [
                    "- The length of your list is incorrect.",
                    f"Your list length is {len(student_answer)}.",
                    "The length should be {len(instructor_answer)}.",
                ]
            )
        else:
            msg_list.append("- The length of the list is correct.")

    if status:
        for s_arr in instructor_answer:
            if not isinstance(s_arr, type(np.zeros(1))):
                status = False
                msg_list.append(
                    "- Element {i} of your list should be of type 'numpy.array'."
                )

    if status:
        msg_list.append("- All list elements are type ndarray.")

    return status, "\n".join(msg_list)


# ======================================================================


# <<<<<<< NOT IN type_handlers >>>>>>
# def check_answer_set_ndarray(student_answer, instructor_answer, rel_tol):
#     """
#     tol: max relative error on the L2 norm
#     Check that all elements in the list have matching norms. Order does not
#     matter. So create a set of norms and match them.
#     """
#     status = True
#     msg_list = []

#     s_answ = list(student_answer)
#     i_answ = list(instructor_answer)

#     for i, arr in enumerate(s_answ):
#         s_answ[i] = np.linalg.norm(arr)
#     for i, arr in enumerate(i_answ):
#         i_answ[i] = np.linalg.norm(arr)

#     s_answ = set(s_answ)
#     i_answ = set(i_answ)

#     status = are_sets_equal(s_answ, i_answ, rtol=rel_tol)
#     if not status:
#         msg_list.append("The arrays can be in any order")

#     return return_value(status, msg_list, s_answ, i_answ)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


# def check_structure_set_ndarray(student_answer):
#     return Tr


# ======================================================================


def check_answer_ndarray(student_answer, instructor_answer, rel_tol):
    """
    rel_tol: max relative error on the L2 norm
    Check that all elements in the list have matching norms
    """
    msg_list = []
    status = True
    s_norm = np.linalg.norm(student_answer)
    i_norm = np.linalg.norm(instructor_answer)

    # Can i_norm be zero?
    status, msg_ = check_float(i_norm, s_norm, rel_tol, 1.0e-5)

    if not status:
        msg_list.append(msg_)
        msg_list.append("For comparison, the array was replaced by its norm")
        msg_list.append(f"The norms have relative error > {rel_tol}")

    return return_value(status, msg_list, s_norm, i_norm)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_ndarray(student_answer):
    """
    Check that all elements in the list have matching norms
    instructor_answer: not used
    """
    if not isinstance(student_answer, type(np.zeros([1]))):
        return (
            False,
            f"- Answer should be a numpy array rather than {type(student_answer)}",
        )
    return True, "Type 'ndarray' is correct."


# ======================================================================


def check_answer_function(student_answer, instructor_answer):
    """
    Student and instructor functions. I will print out the source.
    Ideally, we'd check the arguments, and check the execution of the
    function.
    """
    s_source = inspect.getsource(student_answer)
    i_source = inspect.getsource(instructor_answer)

    msg_list = []
    msg_list.append("Functions are not graded, unless not present.")
    msg_list.append("Instructor function Source")
    msg_list.append(s_source)

    status = True
    return return_value(status, msg_list, s_source, i_source)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_function(student_answer):
    if not isinstance(student_answer, type(lambda: None)):
        return False, "- Answer should be a Python function."
    return True, "Type 'function' is correct."


# ======================================================================


def check_answer_list_list_float(
    student_answer, instructor_answer, rel_tol, partial_score_frac: list[float]
):
    """
    Check two lists of lists of floats with each other
    """
    status = True
    msg_list = []
    ps_dict = init_partial_score_dict()

    for s_lst, i_lst in zip(student_answer, instructor_answer):
        status_, msg_list_ = check_list_float(i_lst, s_lst, rel_tol, 1.0e-6, ps_dict)
        msg_list.extend(msg_list_)
        if status is True:
            status = status_

    partial_score_frac[0] = 1.0 - ps_dict["nb_mismatches"] / ps_dict["nb_total"]
    msg_list.append(f"Answer correct if relative error < {rel_tol*100} percent")
    return return_value(status, msg_list, student_answer, instructor_answer)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_list_list_float(student_answer, instructor_answer):
    """
    Check structure of student_answer.
    instructor_answer: not used
    """
    msg_list = []
    status = True

    if not isinstance(student_answer, list):
        msg_list.append("- answer should be a list.")
        status = False
        return status, "\n".join(msg_list)

    if not len(student_answer) == len(instructor_answer):
        msg_list.append("- Number of elements in the answer is incorrect.")
        status = False

    for i, s_list in enumerate(student_answer):
        if not isinstance(s_list, list):
            msg_list.append(f"- answer[{i}] is not a list. Recheck all elements.")
            status = False
            continue

        for j, el in enumerate(s_list):
            if not isinstance(float(el), float):
                msg_list.append(
                    f"- answer[{i}][{j}] cannot be cast to a float. All elements must be castable to float."
                )
                status = False

    if status:
        msg_list.append("Type 'list[list[float]]' is correct.")

    return status, "\n".join(msg_list)


# ======================================================================


def check_answer_list_set(student_answer, instructor_answer):
    """
    Check two lists of sets (although the set is encoded as a list) with each other

    Both answers are list of sets or lists of list
    """
    msg_list = []
    status = True
    # print("Assert_utilities, type list_set NOT HANDLED")
    return False, "\n".join(msg_list)

    # for s_lst, i_lst in zip(student_answer, instructor_answer):
    #     s_set = set(s_lst)
    #     i_set = set(i_lst)
    #     status = True if i_set == s_set else False

    # msg_list.append(
    #     "Answer is a list of sets; sets can be replaced by lists or tuples."
    # )
    # return return_value(status, msg_list, student_answer, instructor_answer)


# ======================================================================


def convert_to_set_of_sets(input_sequence):
    """
    Convert each inner sequence to a set, then the outer sequence to a set of sets
    """
    return {frozenset(inner) for inner in input_sequence}


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_answer_set_set_int(student_answer, instructor_answer):
    """
    Both student answer and instructor answer should be a set of sets or a structure
    that can be converted to a set of setsr
    """
    status = True
    msg_list = []

    seq_s = student_answer
    seq_i = instructor_answer
    # Convert both sequences to sets of sets
    # One might start with a list of lists [of objects]
    set_of_sets_s = convert_to_set_of_sets(seq_s)
    set_of_sets_i = convert_to_set_of_sets(seq_i)

    #print("set_of_sets_s= ", set_of_sets_s)
    #print("set_of_sets_i= ", set_of_sets_i)

    # Compare the sets of sets
    # What is actually compared?
    status = True if set_of_sets_s == set_of_sets_i else False
    return return_value(status, msg_list, set_of_sets_s, set_of_sets_i)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_set_set_int(student_answer):
    """
    Created by GPT-4, modified by GE, 2024-03-06
    Both student answer and instructor answer should be a set of sets or a structure
    that can be converted to a set of setsr
    Not verified.
    """
    msg_list = []
    status = True
    seq_s = student_answer

    # Function to check if an object is a sequence but not a string
    def is_sequence_but_not_str(obj):
        return isinstance(obj, (list, tuple, set)) and not isinstance(obj, str)

    # Check if the outer structures are sequences
    if not is_sequence_but_not_str(seq_s):
        msg_list.append(
            "- The outer structure is not a sequence (list or set or tuple)."
        )
        status = False
    else:
        msg_list.append("- The outer structure is a sequence (list or set or tuple).")

    # If outer structures are sequences, check each inner structure
    if status:
        for i, seq in enumerate(seq_s, start=1):
            if not is_sequence_but_not_str(seq):
                msg_list.append(
                    "Element {i} of the outer set is not compatible with a set and has type {seq}."
                )
                status = False
                continue
        if status:
            msg_list.append(
                "- All elements of the outer set are compatible with a set (which means I can coerce it into a set"
            )
            msg_list.append("- Answer has the correct structure")

    return status, "\n".join(msg_list)


# ======================================================================


def check_answer_dict_str_tuple_ndarray(
    student_answer, instructor_answer, rel_tol, partial_score_frac: list[float]
):
    """
    GE original function restructed by GPT-4 (2024-03-06)
    student_answer: dictionary with keys:str, values: tuple of ndarrays
    instructor_answer: dictionary with keys:str, values: a set of objects
    rel_tol: tolerance on the matrix norm
    """
    msg_list = []
    status = True  # Assuming correct until proven otherwise
    ps_dict = init_partial_score_dict()

    # Dictionaries to hold norms for student and instructor answers
    s_norms = {}
    i_norms = {}

    for k in instructor_answer.keys():
        # Initialize norms list for current key in both dicts
        s_norms[k] = []
        i_norms[k] = []

        try:
            s_tuple = student_answer[k]
        except KeyError:
            msg_list.append(f"Error: key {repr(k)} is missing")
            continue

        i_tuple = instructor_answer[k]
        for s_arr, i_arr in zip(s_tuple, i_tuple):
            # Calculate norms
            s_norm = np.linalg.norm(s_arr)
            i_norm = np.linalg.norm(i_arr)

            # Store norms
            s_norms[k].append(s_norm)
            i_norms[k].append(i_norm)

        # print(f"{i_norms=}, {s_norms=}")
        status_, msg_ = check_list_float(i_norms[k], s_norms[k], rel_tol, abs_tol=1.e-6, ps_dict=ps_dict)

        if status_ is False:
            msg_list.append(msg_)
            status = False

    partial_score_frac[0] = 1.0 - ps_dict["nb_mismatches"] / ps_dict["nb_total"]
    msg_list.append("Only print the norms of the arrays")
    return return_value(status, msg_list, s_norms, i_norms)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_dict_str_tuple_ndarray(student_answer, instructor_answer):
    status = True
    msg_list = []
    for k, v in instructor_answer.items():
        # repr adds additional quotes; str does not.
        if k not in student_answer:
            msg_list.append(f"- Missing key {repr(k)}")
            status = False
            continue
        if not isinstance(v, (tuple, list)):
            msg_list.append(f"- dict[{repr(k)}] is not a tuple")
            status = False
            continue
        for i, v_el in enumerate(v):
            if not isinstance(v_el, type(np.zeros(1))):
                msg_list.append(f"- dict[{repr(k)}][{i}] is not an numpy array")
                status = False

    if status:
        msg_list.append("Type 'dict[str, tuple(ndarray)]' is correct.")

    return status, "\n".join(msg_list)


# ======================================================================


def check_answer_dendrogram(student_dendro, instructor_dendro, rel_tol):
    """
    With help from GPT-4
    Compares two dendrogram dictionaries.

    Args:
        dend1: The first dendrogram dictionary.
        dend2: The second dendrogram dictionary.
        rel_tol: The relative tolerance for coordinate comparison.

    Returns:
        True if the dendrograms are considered equal within the tolerance,
        False otherwise.
    """
    status = True
    msg_list = []

    dend1 = student_dendro
    dend2 = instructor_dendro

    # Might crash if structures don't match. To fix later.
    # Coordinate comparison
    if not np.allclose(dend1["icoord"], dend2["icoord"], rtol=rel_tol):
        status = False
    if not np.allclose(dend1["dcoord"], dend2["dcoord"], rtol=rel_tol):
        status = False

    # Leaf order comparison
    if not np.array_equal(dend1["leaves"], dend2["leaves"]):
        status = False

    # Optional: Color comparison
    if "color_list" in dend1 and "color_list" in dend2:
        if not all(
            np.array_equal(c1, c2)
            for c1, c2 in zip(dend1["color_list"], dend2["color_list"])
        ):
            return False
    return return_value(status, msg_list, student_dendro, instructor_dendro)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_dendrogram(student_dendro):
    """
    Checks if the structure and types of the student's dendrogram dictionary match
    the expected structure from scipy's dendrogram function.

    Args:
        student_dendro: The dendrogram dictionary from a student.
        instructor_dendro: not used.

    Returns:
        A tuple (bool, str): True and an empty string if the structure matches,
        otherwise False and a message indicating the mismatch.
    """

    # Expected keys in a scipy dendrogram
    expected_keys = {"icoord", "dcoord", "leaves", "ivl", "color_list"}
    student_keys = set(student_dendro.keys())

    # status = True
    # msg_list = []

    # Check for missing keys in student dendrogram
    missing_keys = expected_keys - student_keys
    if missing_keys:
        return False, f"Missing keys in student dendrogram: {missing_keys}."

    # For each key, check if the values are lists and have a consistent structure
    for key in expected_keys:
        if (
            key in student_keys
        ):  # Check only if key is present, though missing keys were already caught
            value = student_dendro[key]

            # Ensure value is a list
            if not isinstance(value, list):
                return False, f"Expected a list for key '{key}', found {type(value)}."

            # Specific checks for 'icoord' and 'dcoord' which should contain lists of lists
            if key in ["icoord", "dcoord"]:
                if not all(isinstance(item, list) for item in value):
                    return False, f"Expected a list of lists for key '{key}'."

            # 'leaves' and 'ivl' checks could be added here, such as ensuring 'leaves' contains integers,
            # and 'ivl' contains strings, if necessary for the scope of your validation.

    # If we reach here, the structure is as expected
    return True, "Type Dendrogram has correct structure."


# ======================================================================


def check_answer_int(student_answer, instructor_answer):
    """ """
    status, msg = check_int(instructor_answer, student_answer)
    return return_value(status, [msg], student_answer, instructor_answer)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_int(student_answer):
    """ """
    if not isinstance(student_answer, int):
        status = False
        msg_list = [
            f"Answer must be of type 'int'. Your answer is of type {type(student_answer).__name__}."
        ]
    else:
        status = True
        msg_list = ["Answer is of type 'int' as expected."]

    return status, "\n".join(msg_list)


# ======================================================================


def check_answer_bool(student_answer, instructor_answer):
    msg_list = []
    status = True

    if student_answer != instructor_answer:
        status = False
        msg_list = ["Answer is incorrect."]
    else:
        status = True
        msg_list = ["Answer is correct."]

    return return_value(status, msg_list, student_answer, instructor_answer)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_bool(student_answer):
    if not isinstance(student_answer, bool):
        status = False
        msg_list = [
            f"Answer must be of type 'bool'. Your answer is of type {type(student_answer)}."
        ]
    else:
        status = True
        msg_list = ["Answer is of type 'bool' as expected."]

    return status, "\n".join(msg_list)


# ======================================================================


def check_answer_list_str(
    student_answer,
    instructor_answer,
    include_indices: list[int],
    exclude_indices: list[int],
    partial_score_frac: list[float],
):
    """
    Normalize and compare the two lists
    These lists are of fixed length. Element should not be added to it,
    unlike dictionaries.
    Either include_indices or exclude_indices must be non-empty lists
    """

    msg_list = []
    status = True
    mismatched_strings = []
    ps_dict = init_partial_score_dict()

    normalized_s_answ = [clean_str_answer(s) for s in student_answer]
    normalized_i_answ = [clean_str_answer(s) for s in instructor_answer]

    for i, i_a in enumerate(normalized_i_answ):
        if exclude_indices != [] and include_indices == [] and i in exclude_indices:
            continue
        if include_indices != [] and exclude_indices == [] and i not in include_indices:
            continue

        ps_dict["nb_total"] += 1
        s_a = normalized_s_answ[i]

        if s_a != i_a:
            status = False
            ps_dict["nb_mismatches"] += 1
            mismatched_strings.append(s_a)

    partial_score_frac[0] = 1.0 - ps_dict["nb_mismatches"] / ps_dict["nb_total"]
    # TODO: Explicitly state the indices considered for grading. 
    # msg_list += [f"List elements in position()s {exclude_indices} is/are not graded.\n"]
    # msg_list += [f"Only list elements in position()s {include_indices} is/are not graded.\n"]
    msg_list += [
        f"There is/are {len(mismatched_strings)} mismatched string(s): ({mismatched_strings})."
    ]

    return return_value(status, msg_list, normalized_s_answ, normalized_i_answ)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_list_str(student_answer):
    msg_list = []

    # Function to check if an object is a list of strings
    def is_list_of_strings(obj):
        return isinstance(obj, list) and all(
            isinstance(element, str) for element in obj
        )

    # Check if both sequences are lists of strings
    if not is_list_of_strings(student_answer):
        msg_list.append("Answer must be a list of strings.")
        status = False
    else:
        msg_list.append("Type 'list[str]' is correct")
        status = True

    return status, "\n".join(msg_list)


# ======================================================================


def check_answer_lineplot(student_answer, instructor_answer, rel_tol):
    """
    Lineplots generated by matlab. Check one or multiple lines.
    Check the following:
        For each line plot:
        Number of points on each plot
        Min and max point values in x and y
    """
    status = True
    msg_list = []

    return return_value(status, msg_list, student_answer, instructor_answer)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_lineplot(student_answer):
    """
    Assume use of Matplotlib
    Lineplots generated by matlab. Check one or multiple lines.
    Check whether subplot is used. This should be stipulated in the assignment.
    Check the number of curves
    Check presence of axis labels and title
    Check presence of grid in x and y

    # I might need intructor answer to check the number of plots. Alternatively,
    # include options to allow specified checking, via kwargs. In that way,
    # I can check the instructor answer AND the student answer structurally
    """
    status = True
    msg_list = []

    # print("==> check structure lineplot")
    # get axis object
    # print(f"{student_answer=}")
    """
    s_ax = student_answer[0].gca()
    i_ax = student_answer[0].gca()
    # print(f"{dir(s_ax)=}")
    # print(f"{type(s_ax).__name__=}")
    # s_ax = s_ax.gcf().get_axes()
    # i_ax = instructor_answer[0].gcf().get_axes()

    if len(s_ax) != len(i_ax):
        msg_list.append(f"There should only be {len(i_ax)} plot(s)!")
        status = False

    if type(student_answer).__name__ != "Line2D":
        msg_list.append("Wrong plot type, not created with plt.plot!")
        status = False

    if not status:
        return status, "\n".join(msg_list)

    # What happens if the label is not a sring? None if no label
    s_xlabel = clean_str_answer(s_ax.get_xlabel())  # if s_ax.get_xlabel() else None
    # i_xlabel = clean_str_answer(i_ax.get_xlabel())  # if s_ax.get_xlabel() else None
    s_ylabel = clean_str_answer(s_ax.get_ylabel())  # if s_ax.get_ylabel() else None
    # i_ylabel = clean_str_answer(i_ax.get_ylabel())  # if s_ax.get_ylabel() else None
    s_title = clean_str_answer(s_ax.get_title())  # if s_ax.get_title() else None
    # i_title = clean_str_answer(i_ax.get_title())  # if s_ax.get_title() else None

    if not s_xlabel or not s_ylabel:
        msg_list.append("Either x or y label is missing! Must be there to get a grade.")
        status = False

    if not s_title:
        msg_list.append("The title is missing")
        status = False
    """

    """ USE LATER
    s_xgrid_vis = any(line.getvisible() for line in s_ax.xaxis.get_gridlines())
    i_xgrid_vis = any(line.getvisible() for line in i_ax.xaxis.get_gridlines())
    s_ygrid_vis = any(line.getvisible() for line in s_ax.yaxis.get_gridlines())
    i_ygrid_vis = any(line.getvisible() for line in i_ax.yaxis.get_gridlines())
    """

    """
    ax:  Axes3D
    fig:  Figure
    scat3:  Path3DCollection
    scat2:  PathCollection
    plot:  list
    plot[0]:  Line2D
    """

    return status, "\n".join(msg_list)


# ======================================================================


# FIX to handle 2D
def check_answer_scatterplot2d(student_answer, instructor_answer, rel_tol):
    status = True
    msg_list = []

    s_answ = student_answer
    i_answ = instructor_answer

    # Check for equality
    # x, y, z = s_answ._offsets2d # protected
    x, y, z = s_answ.get_offsets()
    s_x, s_y, s_z = x.data.astype(float), y.data.astype(float), z.astype(float)

    x, y, z = i_answ._offsets3d
    i_x, i_y, i_z = x.data.astype(float), y.data.astype(float), z.astype(float)

    # print(f"==> {i_x=}, {i_y=}, {i_z=}")
    # print(f"==> {s_x=}, {s_y=}, {s_z=}")

    sum_i = np.sum(i_x) + np.sum(i_y) + np.sum(i_z)
    sum_s = np.sum(s_x) + np.sum(s_y) + np.sum(s_z)

    status, msg = check_float(sum_i, sum_s, rel_tol=rel_tol, abs_tol=1.e-5)
    msg_list.append(msg)

    return return_value(status, msg_list, student_answer, instructor_answer)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_scatterplot2d(student_answer):
    from matplotlib.collections import PathCollection

    status = True
    msg_list = []

    s_answ = student_answer

    if not isinstance(student_answer, PathCollection):
        status = False
        msg_list.append(
            "The answer type should be 'PathCollectdion', the type of the output of 'plt.scatter'."
        )

    xy = s_answ.get_offsets()
    #x, y = s_answ.get_offsets()
    #sxsy = x.data.astype(float), y.data.astype(float)
    #s_x, s_y = x.data.astype(float), y.data.astype(float)

    # x, y, z = i_answ._offsets3d
    # i_x, i_y, i_z = x.data.astype(float), y.data.astype(float), z.astype(float)

    """
    if i_x.shape == s_x.shape and i_y.shape == s_y.shape and i_z.shape == s_z.shape:
        status = False
        msg_list.append(f"The number of points ({s_x.shape[0]}) is incorrect")
    """

    return status, "\n".join(msg_list)


# ======================================================================


def check_answer_scatterplot3d(student_answer, instructor_answer, rel_tol):
    status = True
    msg_list = []

    s_answ = student_answer
    i_answ = instructor_answer

    # Check for equality
    x, y, z = s_answ._offsets3d
    s_x, s_y, s_z = x.data.astype(float), y.data.astype(float), z.astype(float)

    x, y, z = i_answ._offsets3d
    i_x, i_y, i_z = x.data.astype(float), y.data.astype(float), z.astype(float)

    # print(f"==> {i_x=}, {i_y=}, {i_z=}")
    # print(f"==> {s_x=}, {s_y=}, {s_z=}")

    sum_i = np.sum(i_x) + np.sum(i_y) + np.sum(i_z)
    sum_s = np.sum(s_x) + np.sum(s_y) + np.sum(s_z)

    status, msg = check_float(sum_i, sum_s, rel_tol=rel_tol, abs_tol=1.e-5)
    msg_list.append(msg)

    return return_value(status, msg_list, student_answer, instructor_answer)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_scatterplot3d(student_answer):
    from matplotlib.collections import PathCollection

    status = True
    msg_list = []

    s_answ = student_answer

    if not isinstance(student_answer, PathCollection):
        status = False
        msg_list.append(
            "The answer type should be 'PathCollectdion', the type of the output of 'plt.scatter'."
        )

    x, y, z = s_answ._offsets3d
    s_x, s_y, s_z = x.data.astype(float), y.data.astype(float), z.astype(float)

    x, y, z = i_answ._offsets3d
    i_x, i_y, i_z = x.data.astype(float), y.data.astype(float), z.astype(float)

    if i_x.shape == s_x.shape and i_y.shape == s_y.shape and i_z.shape == s_z.shape:
        status = False
        msg_list.append(f"The number of points ({s_x.shape[0]}) is incorrect")

    return status, "\n".join(msg_list)


# ======================================================================
