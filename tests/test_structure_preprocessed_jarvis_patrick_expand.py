
from pytest_utils.decorators import max_score, visibility, hide_errors
import assert_utilities  # <<< SHOULD be specified in config
from my_fixtures import *   
#import my_fixtures
import numpy as np
import yaml
# pytest might change the python path. Make sure to import it last. 
# import pytest

with open('type_handlers.yaml', 'r') as f:
    type_handlers = yaml.safe_load(f)
@hide_errors('')
def test_structure_jarvis_patrick_clustering_jarvis_patrick_function_function(run_compute):
    function_name = test_structure_jarvis_patrick_clustering_jarvis_patrick_function_function
    kwargs = {'student_directory': 'student_code_with_answers' , 'instructor_directory': 'instructor_code_with_answers'}
    correct_answer = run_compute('jarvis_patrick_clustering', 'jarvis_patrick_clustering', 'i', **kwargs)
    if 'jarvis_patrick_function' not in correct_answer:
        explanation = "Key: 'jarvis_patrick_function' not found in instructor answer!\n"
        test_structure_jarvis_patrick_clustering_jarvis_patrick_function_function.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['jarvis_patrick_function']
    student_answer = run_compute('jarvis_patrick_clustering', 'jarvis_patrick_clustering', 's', **kwargs)
    if 'jarvis_patrick_function' not in student_answer:
        explanation = "Key: 'jarvis_patrick_function' not found in student answer!\n"
        test_structure_jarvis_patrick_clustering_jarvis_patrick_function_function.explanation = explanation
        assert False
    else:
        student_answer = student_answer['jarvis_patrick_function']
    local_namespace = {}
    exclude_indices = []
    local_namespace['exclude_indices'] = exclude_indices
    include_indices = []
    local_namespace['include_indices'] = include_indices
    rel_tol = 0.01
    abs_tol = 0.01
    str_choices = []
    local_namespace['str_choices'] = str_choices
    local_namespace['rel_tol'] = rel_tol
    local_namespace['abs_tol'] = abs_tol
    keys = None
    msg_structure = "assert_utilities.check_structure_function(student_answer)"
    msg_answer = "assert_utilities.check_answer_function(student_answer, instructor_answer)"
    local_namespace.update({'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'keys':keys})
    choices = []
    local_namespace['choices'] = choices
    answer_type = 'function'
    question_id = 'jarvis_patrick_clustering'
    subquestion_id = 'jarvis_patrick_function'
    partial_score_frac_l = [0.]
    local_namespace['partial_score_frac_l'] = partial_score_frac_l
    function_name.answer_type = answer_type
    function_name.question_id = question_id
    function_name.subquestion_id = subquestion_id
    function_name.partial_score_frac = partial_score_frac_l[0]
    is_success, explanation_structure = eval(msg_structure, {'__builtins__':{}}, local_namespace)
    explanation = '\n'.join(['==Structure tests==:', explanation_structure])
    test_structure_jarvis_patrick_clustering_jarvis_patrick_function_function.explanation = explanation
    assert is_success


@hide_errors('')
def test_structure_jarvis_patrick_clustering_cluster_parameters_dict_lbrack_str_comma_float_rbrack(run_compute):
    function_name = test_structure_jarvis_patrick_clustering_cluster_parameters_dict_lbrack_str_comma_float_rbrack
    kwargs = {'student_directory': 'student_code_with_answers' , 'instructor_directory': 'instructor_code_with_answers'}
    correct_answer = run_compute('jarvis_patrick_clustering', 'jarvis_patrick_clustering', 'i', **kwargs)
    if 'cluster parameters' not in correct_answer:
        explanation = "Key: 'cluster parameters' not found in instructor answer!\n"
        test_structure_jarvis_patrick_clustering_cluster_parameters_dict_lbrack_str_comma_float_rbrack.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['cluster parameters']
    student_answer = run_compute('jarvis_patrick_clustering', 'jarvis_patrick_clustering', 's', **kwargs)
    if 'cluster parameters' not in student_answer:
        explanation = "Key: 'cluster parameters' not found in student answer!\n"
        test_structure_jarvis_patrick_clustering_cluster_parameters_dict_lbrack_str_comma_float_rbrack.explanation = explanation
        assert False
    else:
        student_answer = student_answer['cluster parameters']
    local_namespace = {}
    exclude_indices = []
    local_namespace['exclude_indices'] = exclude_indices
    include_indices = []
    local_namespace['include_indices'] = include_indices
    rel_tol = 0.01
    abs_tol = 0.01
    str_choices = []
    local_namespace['str_choices'] = str_choices
    dict_float_choices = {}
    local_namespace['dict_float_choices'] = dict_float_choices
    remove_spaces = False
    local_namespace['remove_spaces'] = remove_spaces
    local_namespace['rel_tol'] = rel_tol
    local_namespace['abs_tol'] = abs_tol
    keys = None
    msg_structure = "assert_utilities.check_structure_dict_str_float(student_answer, instructor_answer, keys)"
    msg_answer = "assert_utilities.check_answer_dict_str_float(student_answer, instructor_answer, rel_tol, keys, dict_float_choices, partial_score_frac_l)"
    local_namespace.update({'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'keys':keys})
    choices = []
    local_namespace['choices'] = choices
    answer_type = 'dict[str,float]'
    question_id = 'jarvis_patrick_clustering'
    subquestion_id = 'cluster parameters'
    partial_score_frac_l = [0.]
    local_namespace['partial_score_frac_l'] = partial_score_frac_l
    function_name.answer_type = answer_type
    function_name.question_id = question_id
    function_name.subquestion_id = subquestion_id
    function_name.partial_score_frac = partial_score_frac_l[0]
    is_success, explanation_structure = eval(msg_structure, {'__builtins__':{}}, local_namespace)
    explanation = '\n'.join(['==Structure tests==:', explanation_structure])
    test_structure_jarvis_patrick_clustering_cluster_parameters_dict_lbrack_str_comma_float_rbrack.explanation = explanation
    assert is_success


@hide_errors('')
def test_structure_jarvis_patrick_clustering_1st_group_comma_SSE_dict_lbrack_str_comma_float_rbrack(run_compute):
    function_name = test_structure_jarvis_patrick_clustering_1st_group_comma_SSE_dict_lbrack_str_comma_float_rbrack
    kwargs = {'student_directory': 'student_code_with_answers' , 'instructor_directory': 'instructor_code_with_answers'}
    correct_answer = run_compute('jarvis_patrick_clustering', 'jarvis_patrick_clustering', 'i', **kwargs)
    if '1st group, SSE' not in correct_answer:
        explanation = "Key: '1st group, SSE' not found in instructor answer!\n"
        test_structure_jarvis_patrick_clustering_1st_group_comma_SSE_dict_lbrack_str_comma_float_rbrack.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['1st group, SSE']
    student_answer = run_compute('jarvis_patrick_clustering', 'jarvis_patrick_clustering', 's', **kwargs)
    if '1st group, SSE' not in student_answer:
        explanation = "Key: '1st group, SSE' not found in student answer!\n"
        test_structure_jarvis_patrick_clustering_1st_group_comma_SSE_dict_lbrack_str_comma_float_rbrack.explanation = explanation
        assert False
    else:
        student_answer = student_answer['1st group, SSE']
    local_namespace = {}
    exclude_indices = []
    local_namespace['exclude_indices'] = exclude_indices
    include_indices = []
    local_namespace['include_indices'] = include_indices
    rel_tol = 0.01
    abs_tol = 0.01
    str_choices = []
    local_namespace['str_choices'] = str_choices
    dict_float_choices = {}
    local_namespace['dict_float_choices'] = dict_float_choices
    remove_spaces = False
    local_namespace['remove_spaces'] = remove_spaces
    local_namespace['rel_tol'] = rel_tol
    local_namespace['abs_tol'] = abs_tol
    keys = None
    msg_structure = "assert_utilities.check_structure_dict_str_float(student_answer, instructor_answer, keys)"
    msg_answer = "assert_utilities.check_answer_dict_str_float(student_answer, instructor_answer, rel_tol, keys, dict_float_choices, partial_score_frac_l)"
    local_namespace.update({'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'keys':keys})
    choices = []
    local_namespace['choices'] = choices
    answer_type = 'dict[str,float]'
    question_id = 'jarvis_patrick_clustering'
    subquestion_id = '1st group, SSE'
    partial_score_frac_l = [0.]
    local_namespace['partial_score_frac_l'] = partial_score_frac_l
    function_name.answer_type = answer_type
    function_name.question_id = question_id
    function_name.subquestion_id = subquestion_id
    function_name.partial_score_frac = partial_score_frac_l[0]
    is_success, explanation_structure = eval(msg_structure, {'__builtins__':{}}, local_namespace)
    explanation = '\n'.join(['==Structure tests==:', explanation_structure])
    test_structure_jarvis_patrick_clustering_1st_group_comma_SSE_dict_lbrack_str_comma_float_rbrack.explanation = explanation
    assert is_success


@hide_errors('')
def test_structure_jarvis_patrick_clustering_cluster_scatterplot_with_largest_ARI_scatterplot2d(run_compute):
    function_name = test_structure_jarvis_patrick_clustering_cluster_scatterplot_with_largest_ARI_scatterplot2d
    kwargs = {'student_directory': 'student_code_with_answers' , 'instructor_directory': 'instructor_code_with_answers'}
    correct_answer = run_compute('jarvis_patrick_clustering', 'jarvis_patrick_clustering', 'i', **kwargs)
    if 'cluster scatterplot with largest ARI' not in correct_answer:
        explanation = "Key: 'cluster scatterplot with largest ARI' not found in instructor answer!\n"
        test_structure_jarvis_patrick_clustering_cluster_scatterplot_with_largest_ARI_scatterplot2d.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['cluster scatterplot with largest ARI']
    student_answer = run_compute('jarvis_patrick_clustering', 'jarvis_patrick_clustering', 's', **kwargs)
    if 'cluster scatterplot with largest ARI' not in student_answer:
        explanation = "Key: 'cluster scatterplot with largest ARI' not found in student answer!\n"
        test_structure_jarvis_patrick_clustering_cluster_scatterplot_with_largest_ARI_scatterplot2d.explanation = explanation
        assert False
    else:
        student_answer = student_answer['cluster scatterplot with largest ARI']
    local_namespace = {}
    exclude_indices = []
    local_namespace['exclude_indices'] = exclude_indices
    include_indices = []
    local_namespace['include_indices'] = include_indices
    rel_tol = 0.01
    abs_tol = 0.01
    str_choices = []
    local_namespace['str_choices'] = str_choices
    local_namespace['rel_tol'] = rel_tol
    local_namespace['abs_tol'] = abs_tol
    keys = None
    msg_structure = "assert_utilities.check_structure_scatterplot2d(student_answer)"
    msg_answer = "assert_utilities.check_answer_scatterplot2d(student_answer, instructor_answer, rel_tol)"
    local_namespace.update({'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'keys':keys})
    choices = []
    local_namespace['choices'] = choices
    answer_type = 'scatterplot2d'
    question_id = 'jarvis_patrick_clustering'
    subquestion_id = 'cluster scatterplot with largest ARI'
    partial_score_frac_l = [0.]
    local_namespace['partial_score_frac_l'] = partial_score_frac_l
    function_name.answer_type = answer_type
    function_name.question_id = question_id
    function_name.subquestion_id = subquestion_id
    function_name.partial_score_frac = partial_score_frac_l[0]
    is_success, explanation_structure = eval(msg_structure, {'__builtins__':{}}, local_namespace)
    explanation = '\n'.join(['==Structure tests==:', explanation_structure])
    test_structure_jarvis_patrick_clustering_cluster_scatterplot_with_largest_ARI_scatterplot2d.explanation = explanation
    assert is_success


@hide_errors('')
def test_structure_jarvis_patrick_clustering_cluster_scatterplot_with_smallest_SSE_scatterplot2d(run_compute):
    function_name = test_structure_jarvis_patrick_clustering_cluster_scatterplot_with_smallest_SSE_scatterplot2d
    kwargs = {'student_directory': 'student_code_with_answers' , 'instructor_directory': 'instructor_code_with_answers'}
    correct_answer = run_compute('jarvis_patrick_clustering', 'jarvis_patrick_clustering', 'i', **kwargs)
    if 'cluster scatterplot with smallest SSE' not in correct_answer:
        explanation = "Key: 'cluster scatterplot with smallest SSE' not found in instructor answer!\n"
        test_structure_jarvis_patrick_clustering_cluster_scatterplot_with_smallest_SSE_scatterplot2d.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['cluster scatterplot with smallest SSE']
    student_answer = run_compute('jarvis_patrick_clustering', 'jarvis_patrick_clustering', 's', **kwargs)
    if 'cluster scatterplot with smallest SSE' not in student_answer:
        explanation = "Key: 'cluster scatterplot with smallest SSE' not found in student answer!\n"
        test_structure_jarvis_patrick_clustering_cluster_scatterplot_with_smallest_SSE_scatterplot2d.explanation = explanation
        assert False
    else:
        student_answer = student_answer['cluster scatterplot with smallest SSE']
    local_namespace = {}
    exclude_indices = []
    local_namespace['exclude_indices'] = exclude_indices
    include_indices = []
    local_namespace['include_indices'] = include_indices
    rel_tol = 0.01
    abs_tol = 0.01
    str_choices = []
    local_namespace['str_choices'] = str_choices
    local_namespace['rel_tol'] = rel_tol
    local_namespace['abs_tol'] = abs_tol
    keys = None
    msg_structure = "assert_utilities.check_structure_scatterplot2d(student_answer)"
    msg_answer = "assert_utilities.check_answer_scatterplot2d(student_answer, instructor_answer, rel_tol)"
    local_namespace.update({'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'keys':keys})
    choices = []
    local_namespace['choices'] = choices
    answer_type = 'scatterplot2d'
    question_id = 'jarvis_patrick_clustering'
    subquestion_id = 'cluster scatterplot with smallest SSE'
    partial_score_frac_l = [0.]
    local_namespace['partial_score_frac_l'] = partial_score_frac_l
    function_name.answer_type = answer_type
    function_name.question_id = question_id
    function_name.subquestion_id = subquestion_id
    function_name.partial_score_frac = partial_score_frac_l[0]
    is_success, explanation_structure = eval(msg_structure, {'__builtins__':{}}, local_namespace)
    explanation = '\n'.join(['==Structure tests==:', explanation_structure])
    test_structure_jarvis_patrick_clustering_cluster_scatterplot_with_smallest_SSE_scatterplot2d.explanation = explanation
    assert is_success


@hide_errors('')
def test_structure_jarvis_patrick_clustering_mean_ARIs_float(run_compute):
    function_name = test_structure_jarvis_patrick_clustering_mean_ARIs_float
    kwargs = {'student_directory': 'student_code_with_answers' , 'instructor_directory': 'instructor_code_with_answers'}
    correct_answer = run_compute('jarvis_patrick_clustering', 'jarvis_patrick_clustering', 'i', **kwargs)
    if 'mean_ARIs' not in correct_answer:
        explanation = "Key: 'mean_ARIs' not found in instructor answer!\n"
        test_structure_jarvis_patrick_clustering_mean_ARIs_float.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['mean_ARIs']
    student_answer = run_compute('jarvis_patrick_clustering', 'jarvis_patrick_clustering', 's', **kwargs)
    if 'mean_ARIs' not in student_answer:
        explanation = "Key: 'mean_ARIs' not found in student answer!\n"
        test_structure_jarvis_patrick_clustering_mean_ARIs_float.explanation = explanation
        assert False
    else:
        student_answer = student_answer['mean_ARIs']
    local_namespace = {}
    exclude_indices = []
    local_namespace['exclude_indices'] = exclude_indices
    include_indices = []
    local_namespace['include_indices'] = include_indices
    rel_tol = 0.01
    abs_tol = 0.01
    str_choices = []
    local_namespace['str_choices'] = str_choices
    local_namespace['rel_tol'] = rel_tol
    local_namespace['abs_tol'] = abs_tol
    keys = None
    msg_structure = "assert_utilities.check_structure_float(student_answer)"
    msg_answer = "assert_utilities.check_answer_float(student_answer, instructor_answer, rel_tol, abs_tol)"
    local_namespace.update({'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'keys':keys})
    choices = []
    local_namespace['choices'] = choices
    answer_type = 'float'
    question_id = 'jarvis_patrick_clustering'
    subquestion_id = 'mean_ARIs'
    partial_score_frac_l = [0.]
    local_namespace['partial_score_frac_l'] = partial_score_frac_l
    function_name.answer_type = answer_type
    function_name.question_id = question_id
    function_name.subquestion_id = subquestion_id
    function_name.partial_score_frac = partial_score_frac_l[0]
    is_success, explanation_structure = eval(msg_structure, {'__builtins__':{}}, local_namespace)
    explanation = '\n'.join(['==Structure tests==:', explanation_structure])
    test_structure_jarvis_patrick_clustering_mean_ARIs_float.explanation = explanation
    assert is_success


@hide_errors('')
def test_structure_jarvis_patrick_clustering_std_ARIs_float(run_compute):
    function_name = test_structure_jarvis_patrick_clustering_std_ARIs_float
    kwargs = {'student_directory': 'student_code_with_answers' , 'instructor_directory': 'instructor_code_with_answers'}
    correct_answer = run_compute('jarvis_patrick_clustering', 'jarvis_patrick_clustering', 'i', **kwargs)
    if 'std_ARIs' not in correct_answer:
        explanation = "Key: 'std_ARIs' not found in instructor answer!\n"
        test_structure_jarvis_patrick_clustering_std_ARIs_float.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['std_ARIs']
    student_answer = run_compute('jarvis_patrick_clustering', 'jarvis_patrick_clustering', 's', **kwargs)
    if 'std_ARIs' not in student_answer:
        explanation = "Key: 'std_ARIs' not found in student answer!\n"
        test_structure_jarvis_patrick_clustering_std_ARIs_float.explanation = explanation
        assert False
    else:
        student_answer = student_answer['std_ARIs']
    local_namespace = {}
    exclude_indices = []
    local_namespace['exclude_indices'] = exclude_indices
    include_indices = []
    local_namespace['include_indices'] = include_indices
    rel_tol = 0.01
    abs_tol = 0.01
    str_choices = []
    local_namespace['str_choices'] = str_choices
    local_namespace['rel_tol'] = rel_tol
    local_namespace['abs_tol'] = abs_tol
    keys = None
    msg_structure = "assert_utilities.check_structure_float(student_answer)"
    msg_answer = "assert_utilities.check_answer_float(student_answer, instructor_answer, rel_tol, abs_tol)"
    local_namespace.update({'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'keys':keys})
    choices = []
    local_namespace['choices'] = choices
    answer_type = 'float'
    question_id = 'jarvis_patrick_clustering'
    subquestion_id = 'std_ARIs'
    partial_score_frac_l = [0.]
    local_namespace['partial_score_frac_l'] = partial_score_frac_l
    function_name.answer_type = answer_type
    function_name.question_id = question_id
    function_name.subquestion_id = subquestion_id
    function_name.partial_score_frac = partial_score_frac_l[0]
    is_success, explanation_structure = eval(msg_structure, {'__builtins__':{}}, local_namespace)
    explanation = '\n'.join(['==Structure tests==:', explanation_structure])
    test_structure_jarvis_patrick_clustering_std_ARIs_float.explanation = explanation
    assert is_success


@hide_errors('')
def test_structure_jarvis_patrick_clustering_mean_SSEs_float(run_compute):
    function_name = test_structure_jarvis_patrick_clustering_mean_SSEs_float
    kwargs = {'student_directory': 'student_code_with_answers' , 'instructor_directory': 'instructor_code_with_answers'}
    correct_answer = run_compute('jarvis_patrick_clustering', 'jarvis_patrick_clustering', 'i', **kwargs)
    if 'mean_SSEs' not in correct_answer:
        explanation = "Key: 'mean_SSEs' not found in instructor answer!\n"
        test_structure_jarvis_patrick_clustering_mean_SSEs_float.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['mean_SSEs']
    student_answer = run_compute('jarvis_patrick_clustering', 'jarvis_patrick_clustering', 's', **kwargs)
    if 'mean_SSEs' not in student_answer:
        explanation = "Key: 'mean_SSEs' not found in student answer!\n"
        test_structure_jarvis_patrick_clustering_mean_SSEs_float.explanation = explanation
        assert False
    else:
        student_answer = student_answer['mean_SSEs']
    local_namespace = {}
    exclude_indices = []
    local_namespace['exclude_indices'] = exclude_indices
    include_indices = []
    local_namespace['include_indices'] = include_indices
    rel_tol = 0.01
    abs_tol = 0.01
    str_choices = []
    local_namespace['str_choices'] = str_choices
    local_namespace['rel_tol'] = rel_tol
    local_namespace['abs_tol'] = abs_tol
    keys = None
    msg_structure = "assert_utilities.check_structure_float(student_answer)"
    msg_answer = "assert_utilities.check_answer_float(student_answer, instructor_answer, rel_tol, abs_tol)"
    local_namespace.update({'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'keys':keys})
    choices = []
    local_namespace['choices'] = choices
    answer_type = 'float'
    question_id = 'jarvis_patrick_clustering'
    subquestion_id = 'mean_SSEs'
    partial_score_frac_l = [0.]
    local_namespace['partial_score_frac_l'] = partial_score_frac_l
    function_name.answer_type = answer_type
    function_name.question_id = question_id
    function_name.subquestion_id = subquestion_id
    function_name.partial_score_frac = partial_score_frac_l[0]
    is_success, explanation_structure = eval(msg_structure, {'__builtins__':{}}, local_namespace)
    explanation = '\n'.join(['==Structure tests==:', explanation_structure])
    test_structure_jarvis_patrick_clustering_mean_SSEs_float.explanation = explanation
    assert is_success


@hide_errors('')
def test_structure_jarvis_patrick_clustering_std_SSEs_float(run_compute):
    function_name = test_structure_jarvis_patrick_clustering_std_SSEs_float
    kwargs = {'student_directory': 'student_code_with_answers' , 'instructor_directory': 'instructor_code_with_answers'}
    correct_answer = run_compute('jarvis_patrick_clustering', 'jarvis_patrick_clustering', 'i', **kwargs)
    if 'std_SSEs' not in correct_answer:
        explanation = "Key: 'std_SSEs' not found in instructor answer!\n"
        test_structure_jarvis_patrick_clustering_std_SSEs_float.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['std_SSEs']
    student_answer = run_compute('jarvis_patrick_clustering', 'jarvis_patrick_clustering', 's', **kwargs)
    if 'std_SSEs' not in student_answer:
        explanation = "Key: 'std_SSEs' not found in student answer!\n"
        test_structure_jarvis_patrick_clustering_std_SSEs_float.explanation = explanation
        assert False
    else:
        student_answer = student_answer['std_SSEs']
    local_namespace = {}
    exclude_indices = []
    local_namespace['exclude_indices'] = exclude_indices
    include_indices = []
    local_namespace['include_indices'] = include_indices
    rel_tol = 0.01
    abs_tol = 0.01
    str_choices = []
    local_namespace['str_choices'] = str_choices
    local_namespace['rel_tol'] = rel_tol
    local_namespace['abs_tol'] = abs_tol
    keys = None
    msg_structure = "assert_utilities.check_structure_float(student_answer)"
    msg_answer = "assert_utilities.check_answer_float(student_answer, instructor_answer, rel_tol, abs_tol)"
    local_namespace.update({'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'keys':keys})
    choices = []
    local_namespace['choices'] = choices
    answer_type = 'float'
    question_id = 'jarvis_patrick_clustering'
    subquestion_id = 'std_SSEs'
    partial_score_frac_l = [0.]
    local_namespace['partial_score_frac_l'] = partial_score_frac_l
    function_name.answer_type = answer_type
    function_name.question_id = question_id
    function_name.subquestion_id = subquestion_id
    function_name.partial_score_frac = partial_score_frac_l[0]
    is_success, explanation_structure = eval(msg_structure, {'__builtins__':{}}, local_namespace)
    explanation = '\n'.join(['==Structure tests==:', explanation_structure])
    test_structure_jarvis_patrick_clustering_std_SSEs_float.explanation = explanation
    assert is_success


