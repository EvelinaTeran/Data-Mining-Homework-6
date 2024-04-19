
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

@max_score(20)
@hide_errors('')
def test_answers_gaussian_mixture_plot_original_cluster_scatterplot2d(run_compute):
    function_name = test_answers_gaussian_mixture_plot_original_cluster_scatterplot2d
    kwargs = {'student_directory': 'student_code_with_answers' , 'instructor_directory': 'instructor_code_with_answers'}
    correct_answer = run_compute('expectation_maximization', 'gaussian_mixture', 'i', **kwargs)
    if 'plot_original_cluster' not in correct_answer:
        explanation = "Key: 'plot_original_cluster' not found in instructor answer!\n"
        test_answers_gaussian_mixture_plot_original_cluster_scatterplot2d.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['plot_original_cluster']
    student_answer = run_compute('expectation_maximization', 'gaussian_mixture', 's', **kwargs)
    if 'plot_original_cluster' not in student_answer:
        explanation = "Key: 'plot_original_cluster' not found in student answer!\n"
        test_answers_gaussian_mixture_plot_original_cluster_scatterplot2d.explanation = explanation
        assert False
    else:
        student_answer = student_answer['plot_original_cluster']
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
    question_id = 'gaussian_mixture'
    subquestion_id = 'plot_original_cluster'
    partial_score_frac_l = [0.]
    local_namespace['partial_score_frac_l'] = partial_score_frac_l
    function_name.answer_type = answer_type
    function_name.question_id = question_id
    function_name.subquestion_id = subquestion_id
    function_name.partial_score_frac = partial_score_frac_l[0]
    is_success, explanation_structure = eval(msg_structure, {'__builtins__':{}}, local_namespace)
    if is_success:
        is_success, explanation_answer    = eval(msg_answer,    {'__builtins__':{}}, local_namespace)
        if is_success is True:
            function_name.partial_score_frac = 1.0
        else:
            function_name.partial_score_frac = partial_score_frac_l[0]
    else: 
        explanation_answer = 'Failed structural tests, No grade for answer component\n.' 
        explanation_answer += f'Instructor answer: {repr(correct_answer)}\n'
        explanation_answer += f'Student answer: {repr(student_answer)}'
        function_name.partial_score_frac = partial_score_frac_l[0]
    explanation = '\n'.join(['==Structure tests==:', explanation_structure, '==Answer tests==:', explanation_answer])
    test_answers_gaussian_mixture_plot_original_cluster_scatterplot2d.explanation = explanation
    assert is_success



@max_score(20)
@hide_errors('')
def test_answers_gaussian_mixture_em_algorithm_function_function(run_compute):
    function_name = test_answers_gaussian_mixture_em_algorithm_function_function
    kwargs = {'student_directory': 'student_code_with_answers' , 'instructor_directory': 'instructor_code_with_answers'}
    correct_answer = run_compute('expectation_maximization', 'gaussian_mixture', 'i', **kwargs)
    if 'em_algorithm_function' not in correct_answer:
        explanation = "Key: 'em_algorithm_function' not found in instructor answer!\n"
        test_answers_gaussian_mixture_em_algorithm_function_function.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['em_algorithm_function']
    student_answer = run_compute('expectation_maximization', 'gaussian_mixture', 's', **kwargs)
    if 'em_algorithm_function' not in student_answer:
        explanation = "Key: 'em_algorithm_function' not found in student answer!\n"
        test_answers_gaussian_mixture_em_algorithm_function_function.explanation = explanation
        assert False
    else:
        student_answer = student_answer['em_algorithm_function']
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
    question_id = 'gaussian_mixture'
    subquestion_id = 'em_algorithm_function'
    partial_score_frac_l = [0.]
    local_namespace['partial_score_frac_l'] = partial_score_frac_l
    function_name.answer_type = answer_type
    function_name.question_id = question_id
    function_name.subquestion_id = subquestion_id
    function_name.partial_score_frac = partial_score_frac_l[0]
    is_success, explanation_structure = eval(msg_structure, {'__builtins__':{}}, local_namespace)
    if is_success:
        is_success, explanation_answer    = eval(msg_answer,    {'__builtins__':{}}, local_namespace)
        if is_success is True:
            function_name.partial_score_frac = 1.0
        else:
            function_name.partial_score_frac = partial_score_frac_l[0]
    else: 
        explanation_answer = 'Failed structural tests, No grade for answer component\n.' 
        explanation_answer += f'Instructor answer: {repr(correct_answer)}\n'
        explanation_answer += f'Student answer: {repr(student_answer)}'
        function_name.partial_score_frac = partial_score_frac_l[0]
    explanation = '\n'.join(['==Structure tests==:', explanation_structure, '==Answer tests==:', explanation_answer])
    test_answers_gaussian_mixture_em_algorithm_function_function.explanation = explanation
    assert is_success



@max_score(20)
@hide_errors('')
def test_answers_gaussian_mixture_log_likelihood_ndarray(run_compute):
    function_name = test_answers_gaussian_mixture_log_likelihood_ndarray
    kwargs = {'student_directory': 'student_code_with_answers' , 'instructor_directory': 'instructor_code_with_answers'}
    correct_answer = run_compute('expectation_maximization', 'gaussian_mixture', 'i', **kwargs)
    if 'log_likelihood' not in correct_answer:
        explanation = "Key: 'log_likelihood' not found in instructor answer!\n"
        test_answers_gaussian_mixture_log_likelihood_ndarray.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['log_likelihood']
    student_answer = run_compute('expectation_maximization', 'gaussian_mixture', 's', **kwargs)
    if 'log_likelihood' not in student_answer:
        explanation = "Key: 'log_likelihood' not found in student answer!\n"
        test_answers_gaussian_mixture_log_likelihood_ndarray.explanation = explanation
        assert False
    else:
        student_answer = student_answer['log_likelihood']
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
    msg_structure = "assert_utilities.check_structure_ndarray(student_answer)"
    msg_answer = "assert_utilities.check_answer_ndarray(student_answer, instructor_answer, rel_tol)"
    local_namespace.update({'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'keys':keys})
    choices = []
    local_namespace['choices'] = choices
    answer_type = 'ndarray'
    question_id = 'gaussian_mixture'
    subquestion_id = 'log_likelihood'
    partial_score_frac_l = [0.]
    local_namespace['partial_score_frac_l'] = partial_score_frac_l
    function_name.answer_type = answer_type
    function_name.question_id = question_id
    function_name.subquestion_id = subquestion_id
    function_name.partial_score_frac = partial_score_frac_l[0]
    is_success, explanation_structure = eval(msg_structure, {'__builtins__':{}}, local_namespace)
    if is_success:
        is_success, explanation_answer    = eval(msg_answer,    {'__builtins__':{}}, local_namespace)
        if is_success is True:
            function_name.partial_score_frac = 1.0
        else:
            function_name.partial_score_frac = partial_score_frac_l[0]
    else: 
        explanation_answer = 'Failed structural tests, No grade for answer component\n.' 
        explanation_answer += f'Instructor answer: {repr(correct_answer)}\n'
        explanation_answer += f'Student answer: {repr(student_answer)}'
        function_name.partial_score_frac = partial_score_frac_l[0]
    explanation = '\n'.join(['==Structure tests==:', explanation_structure, '==Answer tests==:', explanation_answer])
    test_answers_gaussian_mixture_log_likelihood_ndarray.explanation = explanation
    assert is_success



@max_score(20)
@hide_errors('')
def test_answers_gaussian_mixture_plot_log_likelihood_lineplot(run_compute):
    function_name = test_answers_gaussian_mixture_plot_log_likelihood_lineplot
    kwargs = {'student_directory': 'student_code_with_answers' , 'instructor_directory': 'instructor_code_with_answers'}
    correct_answer = run_compute('expectation_maximization', 'gaussian_mixture', 'i', **kwargs)
    if 'plot_log_likelihood' not in correct_answer:
        explanation = "Key: 'plot_log_likelihood' not found in instructor answer!\n"
        test_answers_gaussian_mixture_plot_log_likelihood_lineplot.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['plot_log_likelihood']
    student_answer = run_compute('expectation_maximization', 'gaussian_mixture', 's', **kwargs)
    if 'plot_log_likelihood' not in student_answer:
        explanation = "Key: 'plot_log_likelihood' not found in student answer!\n"
        test_answers_gaussian_mixture_plot_log_likelihood_lineplot.explanation = explanation
        assert False
    else:
        student_answer = student_answer['plot_log_likelihood']
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
    msg_structure = "assert_utilities.check_structure_lineplot(student_answer)"
    msg_answer = "assert_utilities.check_answer_lineplot(student_answer, instructor_answer, rel_tol)"
    local_namespace.update({'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'keys':keys})
    choices = []
    local_namespace['choices'] = choices
    answer_type = 'lineplot'
    question_id = 'gaussian_mixture'
    subquestion_id = 'plot_log_likelihood'
    partial_score_frac_l = [0.]
    local_namespace['partial_score_frac_l'] = partial_score_frac_l
    function_name.answer_type = answer_type
    function_name.question_id = question_id
    function_name.subquestion_id = subquestion_id
    function_name.partial_score_frac = partial_score_frac_l[0]
    is_success, explanation_structure = eval(msg_structure, {'__builtins__':{}}, local_namespace)
    if is_success:
        is_success, explanation_answer    = eval(msg_answer,    {'__builtins__':{}}, local_namespace)
        if is_success is True:
            function_name.partial_score_frac = 1.0
        else:
            function_name.partial_score_frac = partial_score_frac_l[0]
    else: 
        explanation_answer = 'Failed structural tests, No grade for answer component\n.' 
        explanation_answer += f'Instructor answer: {repr(correct_answer)}\n'
        explanation_answer += f'Student answer: {repr(student_answer)}'
        function_name.partial_score_frac = partial_score_frac_l[0]
    explanation = '\n'.join(['==Structure tests==:', explanation_structure, '==Answer tests==:', explanation_answer])
    test_answers_gaussian_mixture_plot_log_likelihood_lineplot.explanation = explanation
    assert is_success



@max_score(20)
@hide_errors('')
def test_answers_gaussian_mixture_probability_1_mean_list_lbrack_float_rbrack(run_compute):
    function_name = test_answers_gaussian_mixture_probability_1_mean_list_lbrack_float_rbrack
    kwargs = {'student_directory': 'student_code_with_answers' , 'instructor_directory': 'instructor_code_with_answers'}
    correct_answer = run_compute('expectation_maximization', 'gaussian_mixture', 'i', **kwargs)
    if 'probability_1_mean' not in correct_answer:
        explanation = "Key: 'probability_1_mean' not found in instructor answer!\n"
        test_answers_gaussian_mixture_probability_1_mean_list_lbrack_float_rbrack.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['probability_1_mean']
    student_answer = run_compute('expectation_maximization', 'gaussian_mixture', 's', **kwargs)
    if 'probability_1_mean' not in student_answer:
        explanation = "Key: 'probability_1_mean' not found in student answer!\n"
        test_answers_gaussian_mixture_probability_1_mean_list_lbrack_float_rbrack.explanation = explanation
        assert False
    else:
        student_answer = student_answer['probability_1_mean']
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
    msg_structure = "assert_utilities.check_structure_list_float(student_answer, instructor_answer)"
    msg_answer = "assert_utilities.check_answer_list_float(student_answer, instructor_answer, rel_tol, monotone_increasing, partial_score_frac)"
    local_namespace.update({'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'keys':keys})
    choices = []
    local_namespace['choices'] = choices
    answer_type = 'list[float]'
    question_id = 'gaussian_mixture'
    subquestion_id = 'probability_1_mean'
    partial_score_frac_l = [0.]
    local_namespace['partial_score_frac_l'] = partial_score_frac_l
    function_name.answer_type = answer_type
    function_name.question_id = question_id
    function_name.subquestion_id = subquestion_id
    function_name.partial_score_frac = partial_score_frac_l[0]
    is_success, explanation_structure = eval(msg_structure, {'__builtins__':{}}, local_namespace)
    if is_success:
        is_success, explanation_answer    = eval(msg_answer,    {'__builtins__':{}}, local_namespace)
        if is_success is True:
            function_name.partial_score_frac = 1.0
        else:
            function_name.partial_score_frac = partial_score_frac_l[0]
    else: 
        explanation_answer = 'Failed structural tests, No grade for answer component\n.' 
        explanation_answer += f'Instructor answer: {repr(correct_answer)}\n'
        explanation_answer += f'Student answer: {repr(student_answer)}'
        function_name.partial_score_frac = partial_score_frac_l[0]
    explanation = '\n'.join(['==Structure tests==:', explanation_structure, '==Answer tests==:', explanation_answer])
    test_answers_gaussian_mixture_probability_1_mean_list_lbrack_float_rbrack.explanation = explanation
    assert is_success



@max_score(20)
@hide_errors('')
def test_answers_gaussian_mixture_probability_2_mean_list_lbrack_float_rbrack(run_compute):
    function_name = test_answers_gaussian_mixture_probability_2_mean_list_lbrack_float_rbrack
    kwargs = {'student_directory': 'student_code_with_answers' , 'instructor_directory': 'instructor_code_with_answers'}
    correct_answer = run_compute('expectation_maximization', 'gaussian_mixture', 'i', **kwargs)
    if 'probability_2_mean' not in correct_answer:
        explanation = "Key: 'probability_2_mean' not found in instructor answer!\n"
        test_answers_gaussian_mixture_probability_2_mean_list_lbrack_float_rbrack.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['probability_2_mean']
    student_answer = run_compute('expectation_maximization', 'gaussian_mixture', 's', **kwargs)
    if 'probability_2_mean' not in student_answer:
        explanation = "Key: 'probability_2_mean' not found in student answer!\n"
        test_answers_gaussian_mixture_probability_2_mean_list_lbrack_float_rbrack.explanation = explanation
        assert False
    else:
        student_answer = student_answer['probability_2_mean']
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
    msg_structure = "assert_utilities.check_structure_list_float(student_answer, instructor_answer)"
    msg_answer = "assert_utilities.check_answer_list_float(student_answer, instructor_answer, rel_tol, monotone_increasing, partial_score_frac)"
    local_namespace.update({'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'keys':keys})
    choices = []
    local_namespace['choices'] = choices
    answer_type = 'list[float]'
    question_id = 'gaussian_mixture'
    subquestion_id = 'probability_2_mean'
    partial_score_frac_l = [0.]
    local_namespace['partial_score_frac_l'] = partial_score_frac_l
    function_name.answer_type = answer_type
    function_name.question_id = question_id
    function_name.subquestion_id = subquestion_id
    function_name.partial_score_frac = partial_score_frac_l[0]
    is_success, explanation_structure = eval(msg_structure, {'__builtins__':{}}, local_namespace)
    if is_success:
        is_success, explanation_answer    = eval(msg_answer,    {'__builtins__':{}}, local_namespace)
        if is_success is True:
            function_name.partial_score_frac = 1.0
        else:
            function_name.partial_score_frac = partial_score_frac_l[0]
    else: 
        explanation_answer = 'Failed structural tests, No grade for answer component\n.' 
        explanation_answer += f'Instructor answer: {repr(correct_answer)}\n'
        explanation_answer += f'Student answer: {repr(student_answer)}'
        function_name.partial_score_frac = partial_score_frac_l[0]
    explanation = '\n'.join(['==Structure tests==:', explanation_structure, '==Answer tests==:', explanation_answer])
    test_answers_gaussian_mixture_probability_2_mean_list_lbrack_float_rbrack.explanation = explanation
    assert is_success



@max_score(20)
@hide_errors('')
def test_answers_gaussian_mixture_probability_1_covariance_list_lbrack_ndarray_rbrack(run_compute):
    function_name = test_answers_gaussian_mixture_probability_1_covariance_list_lbrack_ndarray_rbrack
    kwargs = {'student_directory': 'student_code_with_answers' , 'instructor_directory': 'instructor_code_with_answers'}
    correct_answer = run_compute('expectation_maximization', 'gaussian_mixture', 'i', **kwargs)
    if 'probability_1_covariance' not in correct_answer:
        explanation = "Key: 'probability_1_covariance' not found in instructor answer!\n"
        test_answers_gaussian_mixture_probability_1_covariance_list_lbrack_ndarray_rbrack.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['probability_1_covariance']
    student_answer = run_compute('expectation_maximization', 'gaussian_mixture', 's', **kwargs)
    if 'probability_1_covariance' not in student_answer:
        explanation = "Key: 'probability_1_covariance' not found in student answer!\n"
        test_answers_gaussian_mixture_probability_1_covariance_list_lbrack_ndarray_rbrack.explanation = explanation
        assert False
    else:
        student_answer = student_answer['probability_1_covariance']
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
    msg_structure = "assert_utilities.check_structure_list_ndarray(student_answer, instructor_answer)"
    msg_answer = "assert_utilities.check_answer_list_ndarray(student_answer, instructor_answer, rel_tol, partial_score_frac_l)"
    local_namespace.update({'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'keys':keys})
    choices = []
    local_namespace['choices'] = choices
    answer_type = 'list[ndarray]'
    question_id = 'gaussian_mixture'
    subquestion_id = 'probability_1_covariance'
    partial_score_frac_l = [0.]
    local_namespace['partial_score_frac_l'] = partial_score_frac_l
    function_name.answer_type = answer_type
    function_name.question_id = question_id
    function_name.subquestion_id = subquestion_id
    function_name.partial_score_frac = partial_score_frac_l[0]
    is_success, explanation_structure = eval(msg_structure, {'__builtins__':{}}, local_namespace)
    if is_success:
        is_success, explanation_answer    = eval(msg_answer,    {'__builtins__':{}}, local_namespace)
        if is_success is True:
            function_name.partial_score_frac = 1.0
        else:
            function_name.partial_score_frac = partial_score_frac_l[0]
    else: 
        explanation_answer = 'Failed structural tests, No grade for answer component\n.' 
        explanation_answer += f'Instructor answer: {repr(correct_answer)}\n'
        explanation_answer += f'Student answer: {repr(student_answer)}'
        function_name.partial_score_frac = partial_score_frac_l[0]
    explanation = '\n'.join(['==Structure tests==:', explanation_structure, '==Answer tests==:', explanation_answer])
    test_answers_gaussian_mixture_probability_1_covariance_list_lbrack_ndarray_rbrack.explanation = explanation
    assert is_success



@max_score(20)
@hide_errors('')
def test_answers_gaussian_mixture_probability_2_covariance_list_lbrack_ndarray_rbrack(run_compute):
    function_name = test_answers_gaussian_mixture_probability_2_covariance_list_lbrack_ndarray_rbrack
    kwargs = {'student_directory': 'student_code_with_answers' , 'instructor_directory': 'instructor_code_with_answers'}
    correct_answer = run_compute('expectation_maximization', 'gaussian_mixture', 'i', **kwargs)
    if 'probability_2_covariance' not in correct_answer:
        explanation = "Key: 'probability_2_covariance' not found in instructor answer!\n"
        test_answers_gaussian_mixture_probability_2_covariance_list_lbrack_ndarray_rbrack.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['probability_2_covariance']
    student_answer = run_compute('expectation_maximization', 'gaussian_mixture', 's', **kwargs)
    if 'probability_2_covariance' not in student_answer:
        explanation = "Key: 'probability_2_covariance' not found in student answer!\n"
        test_answers_gaussian_mixture_probability_2_covariance_list_lbrack_ndarray_rbrack.explanation = explanation
        assert False
    else:
        student_answer = student_answer['probability_2_covariance']
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
    msg_structure = "assert_utilities.check_structure_list_ndarray(student_answer, instructor_answer)"
    msg_answer = "assert_utilities.check_answer_list_ndarray(student_answer, instructor_answer, rel_tol, partial_score_frac_l)"
    local_namespace.update({'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'keys':keys})
    choices = []
    local_namespace['choices'] = choices
    answer_type = 'list[ndarray]'
    question_id = 'gaussian_mixture'
    subquestion_id = 'probability_2_covariance'
    partial_score_frac_l = [0.]
    local_namespace['partial_score_frac_l'] = partial_score_frac_l
    function_name.answer_type = answer_type
    function_name.question_id = question_id
    function_name.subquestion_id = subquestion_id
    function_name.partial_score_frac = partial_score_frac_l[0]
    is_success, explanation_structure = eval(msg_structure, {'__builtins__':{}}, local_namespace)
    if is_success:
        is_success, explanation_answer    = eval(msg_answer,    {'__builtins__':{}}, local_namespace)
        if is_success is True:
            function_name.partial_score_frac = 1.0
        else:
            function_name.partial_score_frac = partial_score_frac_l[0]
    else: 
        explanation_answer = 'Failed structural tests, No grade for answer component\n.' 
        explanation_answer += f'Instructor answer: {repr(correct_answer)}\n'
        explanation_answer += f'Student answer: {repr(student_answer)}'
        function_name.partial_score_frac = partial_score_frac_l[0]
    explanation = '\n'.join(['==Structure tests==:', explanation_structure, '==Answer tests==:', explanation_answer])
    test_answers_gaussian_mixture_probability_2_covariance_list_lbrack_ndarray_rbrack.explanation = explanation
    assert is_success



@max_score(20)
@hide_errors('')
def test_answers_gaussian_mixture_probability_1_amplitude_list_lbrack_float_rbrack(run_compute):
    function_name = test_answers_gaussian_mixture_probability_1_amplitude_list_lbrack_float_rbrack
    kwargs = {'student_directory': 'student_code_with_answers' , 'instructor_directory': 'instructor_code_with_answers'}
    correct_answer = run_compute('expectation_maximization', 'gaussian_mixture', 'i', **kwargs)
    if 'probability_1_amplitude' not in correct_answer:
        explanation = "Key: 'probability_1_amplitude' not found in instructor answer!\n"
        test_answers_gaussian_mixture_probability_1_amplitude_list_lbrack_float_rbrack.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['probability_1_amplitude']
    student_answer = run_compute('expectation_maximization', 'gaussian_mixture', 's', **kwargs)
    if 'probability_1_amplitude' not in student_answer:
        explanation = "Key: 'probability_1_amplitude' not found in student answer!\n"
        test_answers_gaussian_mixture_probability_1_amplitude_list_lbrack_float_rbrack.explanation = explanation
        assert False
    else:
        student_answer = student_answer['probability_1_amplitude']
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
    msg_structure = "assert_utilities.check_structure_list_float(student_answer, instructor_answer)"
    msg_answer = "assert_utilities.check_answer_list_float(student_answer, instructor_answer, rel_tol, monotone_increasing, partial_score_frac)"
    local_namespace.update({'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'keys':keys})
    choices = []
    local_namespace['choices'] = choices
    answer_type = 'list[float]'
    question_id = 'gaussian_mixture'
    subquestion_id = 'probability_1_amplitude'
    partial_score_frac_l = [0.]
    local_namespace['partial_score_frac_l'] = partial_score_frac_l
    function_name.answer_type = answer_type
    function_name.question_id = question_id
    function_name.subquestion_id = subquestion_id
    function_name.partial_score_frac = partial_score_frac_l[0]
    is_success, explanation_structure = eval(msg_structure, {'__builtins__':{}}, local_namespace)
    if is_success:
        is_success, explanation_answer    = eval(msg_answer,    {'__builtins__':{}}, local_namespace)
        if is_success is True:
            function_name.partial_score_frac = 1.0
        else:
            function_name.partial_score_frac = partial_score_frac_l[0]
    else: 
        explanation_answer = 'Failed structural tests, No grade for answer component\n.' 
        explanation_answer += f'Instructor answer: {repr(correct_answer)}\n'
        explanation_answer += f'Student answer: {repr(student_answer)}'
        function_name.partial_score_frac = partial_score_frac_l[0]
    explanation = '\n'.join(['==Structure tests==:', explanation_structure, '==Answer tests==:', explanation_answer])
    test_answers_gaussian_mixture_probability_1_amplitude_list_lbrack_float_rbrack.explanation = explanation
    assert is_success



@max_score(20)
@hide_errors('')
def test_answers_gaussian_mixture_probability_2_amplitude_list_lbrack_float_rbrack(run_compute):
    function_name = test_answers_gaussian_mixture_probability_2_amplitude_list_lbrack_float_rbrack
    kwargs = {'student_directory': 'student_code_with_answers' , 'instructor_directory': 'instructor_code_with_answers'}
    correct_answer = run_compute('expectation_maximization', 'gaussian_mixture', 'i', **kwargs)
    if 'probability_2_amplitude' not in correct_answer:
        explanation = "Key: 'probability_2_amplitude' not found in instructor answer!\n"
        test_answers_gaussian_mixture_probability_2_amplitude_list_lbrack_float_rbrack.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['probability_2_amplitude']
    student_answer = run_compute('expectation_maximization', 'gaussian_mixture', 's', **kwargs)
    if 'probability_2_amplitude' not in student_answer:
        explanation = "Key: 'probability_2_amplitude' not found in student answer!\n"
        test_answers_gaussian_mixture_probability_2_amplitude_list_lbrack_float_rbrack.explanation = explanation
        assert False
    else:
        student_answer = student_answer['probability_2_amplitude']
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
    msg_structure = "assert_utilities.check_structure_list_float(student_answer, instructor_answer)"
    msg_answer = "assert_utilities.check_answer_list_float(student_answer, instructor_answer, rel_tol, monotone_increasing, partial_score_frac)"
    local_namespace.update({'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'keys':keys})
    choices = []
    local_namespace['choices'] = choices
    answer_type = 'list[float]'
    question_id = 'gaussian_mixture'
    subquestion_id = 'probability_2_amplitude'
    partial_score_frac_l = [0.]
    local_namespace['partial_score_frac_l'] = partial_score_frac_l
    function_name.answer_type = answer_type
    function_name.question_id = question_id
    function_name.subquestion_id = subquestion_id
    function_name.partial_score_frac = partial_score_frac_l[0]
    is_success, explanation_structure = eval(msg_structure, {'__builtins__':{}}, local_namespace)
    if is_success:
        is_success, explanation_answer    = eval(msg_answer,    {'__builtins__':{}}, local_namespace)
        if is_success is True:
            function_name.partial_score_frac = 1.0
        else:
            function_name.partial_score_frac = partial_score_frac_l[0]
    else: 
        explanation_answer = 'Failed structural tests, No grade for answer component\n.' 
        explanation_answer += f'Instructor answer: {repr(correct_answer)}\n'
        explanation_answer += f'Student answer: {repr(student_answer)}'
        function_name.partial_score_frac = partial_score_frac_l[0]
    explanation = '\n'.join(['==Structure tests==:', explanation_structure, '==Answer tests==:', explanation_answer])
    test_answers_gaussian_mixture_probability_2_amplitude_list_lbrack_float_rbrack.explanation = explanation
    assert is_success



@max_score(20)
@hide_errors('')
def test_answers_gaussian_mixture_average_confusion_matrix_dict_lbrack_NDArray_rbrack(run_compute):
    function_name = test_answers_gaussian_mixture_average_confusion_matrix_dict_lbrack_NDArray_rbrack
    kwargs = {'student_directory': 'student_code_with_answers' , 'instructor_directory': 'instructor_code_with_answers'}
    correct_answer = run_compute('expectation_maximization', 'gaussian_mixture', 'i', **kwargs)
    if 'average_confusion_matrix' not in correct_answer:
        explanation = "Key: 'average_confusion_matrix' not found in instructor answer!\n"
        test_answers_gaussian_mixture_average_confusion_matrix_dict_lbrack_NDArray_rbrack.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['average_confusion_matrix']
    student_answer = run_compute('expectation_maximization', 'gaussian_mixture', 's', **kwargs)
    if 'average_confusion_matrix' not in student_answer:
        explanation = "Key: 'average_confusion_matrix' not found in student answer!\n"
        test_answers_gaussian_mixture_average_confusion_matrix_dict_lbrack_NDArray_rbrack.explanation = explanation
        assert False
    else:
        student_answer = student_answer['average_confusion_matrix']
    print('type dict[NDArray] NOT HANDLED!')



@max_score(20)
@hide_errors('')
def test_answers_gaussian_mixture_std_confusion_matrix_dict_lbrack_NDArray_rbrack(run_compute):
    function_name = test_answers_gaussian_mixture_std_confusion_matrix_dict_lbrack_NDArray_rbrack
    kwargs = {'student_directory': 'student_code_with_answers' , 'instructor_directory': 'instructor_code_with_answers'}
    correct_answer = run_compute('expectation_maximization', 'gaussian_mixture', 'i', **kwargs)
    if 'std_confusion_matrix' not in correct_answer:
        explanation = "Key: 'std_confusion_matrix' not found in instructor answer!\n"
        test_answers_gaussian_mixture_std_confusion_matrix_dict_lbrack_NDArray_rbrack.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['std_confusion_matrix']
    student_answer = run_compute('expectation_maximization', 'gaussian_mixture', 's', **kwargs)
    if 'std_confusion_matrix' not in student_answer:
        explanation = "Key: 'std_confusion_matrix' not found in student answer!\n"
        test_answers_gaussian_mixture_std_confusion_matrix_dict_lbrack_NDArray_rbrack.explanation = explanation
        assert False
    else:
        student_answer = student_answer['std_confusion_matrix']
    print('type dict[NDArray] NOT HANDLED!')



@max_score(20)
@hide_errors('')
def test_answers_gaussian_mixture_ARI_list_lbrack_float_rbrack(run_compute):
    function_name = test_answers_gaussian_mixture_ARI_list_lbrack_float_rbrack
    kwargs = {'student_directory': 'student_code_with_answers' , 'instructor_directory': 'instructor_code_with_answers'}
    correct_answer = run_compute('expectation_maximization', 'gaussian_mixture', 'i', **kwargs)
    if 'ARI' not in correct_answer:
        explanation = "Key: 'ARI' not found in instructor answer!\n"
        test_answers_gaussian_mixture_ARI_list_lbrack_float_rbrack.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['ARI']
    student_answer = run_compute('expectation_maximization', 'gaussian_mixture', 's', **kwargs)
    if 'ARI' not in student_answer:
        explanation = "Key: 'ARI' not found in student answer!\n"
        test_answers_gaussian_mixture_ARI_list_lbrack_float_rbrack.explanation = explanation
        assert False
    else:
        student_answer = student_answer['ARI']
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
    msg_structure = "assert_utilities.check_structure_list_float(student_answer, instructor_answer)"
    msg_answer = "assert_utilities.check_answer_list_float(student_answer, instructor_answer, rel_tol, monotone_increasing, partial_score_frac)"
    local_namespace.update({'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'keys':keys})
    choices = []
    local_namespace['choices'] = choices
    answer_type = 'list[float]'
    question_id = 'gaussian_mixture'
    subquestion_id = 'ARI'
    partial_score_frac_l = [0.]
    local_namespace['partial_score_frac_l'] = partial_score_frac_l
    function_name.answer_type = answer_type
    function_name.question_id = question_id
    function_name.subquestion_id = subquestion_id
    function_name.partial_score_frac = partial_score_frac_l[0]
    is_success, explanation_structure = eval(msg_structure, {'__builtins__':{}}, local_namespace)
    if is_success:
        is_success, explanation_answer    = eval(msg_answer,    {'__builtins__':{}}, local_namespace)
        if is_success is True:
            function_name.partial_score_frac = 1.0
        else:
            function_name.partial_score_frac = partial_score_frac_l[0]
    else: 
        explanation_answer = 'Failed structural tests, No grade for answer component\n.' 
        explanation_answer += f'Instructor answer: {repr(correct_answer)}\n'
        explanation_answer += f'Student answer: {repr(student_answer)}'
        function_name.partial_score_frac = partial_score_frac_l[0]
    explanation = '\n'.join(['==Structure tests==:', explanation_structure, '==Answer tests==:', explanation_answer])
    test_answers_gaussian_mixture_ARI_list_lbrack_float_rbrack.explanation = explanation
    assert is_success



@max_score(20)
@hide_errors('')
def test_answers_gaussian_mixture_SSE_list_lbrack_float_rbrack(run_compute):
    function_name = test_answers_gaussian_mixture_SSE_list_lbrack_float_rbrack
    kwargs = {'student_directory': 'student_code_with_answers' , 'instructor_directory': 'instructor_code_with_answers'}
    correct_answer = run_compute('expectation_maximization', 'gaussian_mixture', 'i', **kwargs)
    if 'SSE' not in correct_answer:
        explanation = "Key: 'SSE' not found in instructor answer!\n"
        test_answers_gaussian_mixture_SSE_list_lbrack_float_rbrack.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['SSE']
    student_answer = run_compute('expectation_maximization', 'gaussian_mixture', 's', **kwargs)
    if 'SSE' not in student_answer:
        explanation = "Key: 'SSE' not found in student answer!\n"
        test_answers_gaussian_mixture_SSE_list_lbrack_float_rbrack.explanation = explanation
        assert False
    else:
        student_answer = student_answer['SSE']
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
    msg_structure = "assert_utilities.check_structure_list_float(student_answer, instructor_answer)"
    msg_answer = "assert_utilities.check_answer_list_float(student_answer, instructor_answer, rel_tol, monotone_increasing, partial_score_frac)"
    local_namespace.update({'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'keys':keys})
    choices = []
    local_namespace['choices'] = choices
    answer_type = 'list[float]'
    question_id = 'gaussian_mixture'
    subquestion_id = 'SSE'
    partial_score_frac_l = [0.]
    local_namespace['partial_score_frac_l'] = partial_score_frac_l
    function_name.answer_type = answer_type
    function_name.question_id = question_id
    function_name.subquestion_id = subquestion_id
    function_name.partial_score_frac = partial_score_frac_l[0]
    is_success, explanation_structure = eval(msg_structure, {'__builtins__':{}}, local_namespace)
    if is_success:
        is_success, explanation_answer    = eval(msg_answer,    {'__builtins__':{}}, local_namespace)
        if is_success is True:
            function_name.partial_score_frac = 1.0
        else:
            function_name.partial_score_frac = partial_score_frac_l[0]
    else: 
        explanation_answer = 'Failed structural tests, No grade for answer component\n.' 
        explanation_answer += f'Instructor answer: {repr(correct_answer)}\n'
        explanation_answer += f'Student answer: {repr(student_answer)}'
        function_name.partial_score_frac = partial_score_frac_l[0]
    explanation = '\n'.join(['==Structure tests==:', explanation_structure, '==Answer tests==:', explanation_answer])
    test_answers_gaussian_mixture_SSE_list_lbrack_float_rbrack.explanation = explanation
    assert is_success



@max_score(20)
@hide_errors('')
def test_answers_gaussian_mixture_avg_std_ARI_list_lbrack_float_rbrack(run_compute):
    function_name = test_answers_gaussian_mixture_avg_std_ARI_list_lbrack_float_rbrack
    kwargs = {'student_directory': 'student_code_with_answers' , 'instructor_directory': 'instructor_code_with_answers'}
    correct_answer = run_compute('expectation_maximization', 'gaussian_mixture', 'i', **kwargs)
    if 'avg_std_ARI' not in correct_answer:
        explanation = "Key: 'avg_std_ARI' not found in instructor answer!\n"
        test_answers_gaussian_mixture_avg_std_ARI_list_lbrack_float_rbrack.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['avg_std_ARI']
    student_answer = run_compute('expectation_maximization', 'gaussian_mixture', 's', **kwargs)
    if 'avg_std_ARI' not in student_answer:
        explanation = "Key: 'avg_std_ARI' not found in student answer!\n"
        test_answers_gaussian_mixture_avg_std_ARI_list_lbrack_float_rbrack.explanation = explanation
        assert False
    else:
        student_answer = student_answer['avg_std_ARI']
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
    msg_structure = "assert_utilities.check_structure_list_float(student_answer, instructor_answer)"
    msg_answer = "assert_utilities.check_answer_list_float(student_answer, instructor_answer, rel_tol, monotone_increasing, partial_score_frac)"
    local_namespace.update({'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'keys':keys})
    choices = []
    local_namespace['choices'] = choices
    answer_type = 'list[float]'
    question_id = 'gaussian_mixture'
    subquestion_id = 'avg_std_ARI'
    partial_score_frac_l = [0.]
    local_namespace['partial_score_frac_l'] = partial_score_frac_l
    function_name.answer_type = answer_type
    function_name.question_id = question_id
    function_name.subquestion_id = subquestion_id
    function_name.partial_score_frac = partial_score_frac_l[0]
    is_success, explanation_structure = eval(msg_structure, {'__builtins__':{}}, local_namespace)
    if is_success:
        is_success, explanation_answer    = eval(msg_answer,    {'__builtins__':{}}, local_namespace)
        if is_success is True:
            function_name.partial_score_frac = 1.0
        else:
            function_name.partial_score_frac = partial_score_frac_l[0]
    else: 
        explanation_answer = 'Failed structural tests, No grade for answer component\n.' 
        explanation_answer += f'Instructor answer: {repr(correct_answer)}\n'
        explanation_answer += f'Student answer: {repr(student_answer)}'
        function_name.partial_score_frac = partial_score_frac_l[0]
    explanation = '\n'.join(['==Structure tests==:', explanation_structure, '==Answer tests==:', explanation_answer])
    test_answers_gaussian_mixture_avg_std_ARI_list_lbrack_float_rbrack.explanation = explanation
    assert is_success



@max_score(20)
@hide_errors('')
def test_answers_gaussian_mixture_avg_std_SSE_list_lbrack_float_rbrack(run_compute):
    function_name = test_answers_gaussian_mixture_avg_std_SSE_list_lbrack_float_rbrack
    kwargs = {'student_directory': 'student_code_with_answers' , 'instructor_directory': 'instructor_code_with_answers'}
    correct_answer = run_compute('expectation_maximization', 'gaussian_mixture', 'i', **kwargs)
    if 'avg_std_SSE' not in correct_answer:
        explanation = "Key: 'avg_std_SSE' not found in instructor answer!\n"
        test_answers_gaussian_mixture_avg_std_SSE_list_lbrack_float_rbrack.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['avg_std_SSE']
    student_answer = run_compute('expectation_maximization', 'gaussian_mixture', 's', **kwargs)
    if 'avg_std_SSE' not in student_answer:
        explanation = "Key: 'avg_std_SSE' not found in student answer!\n"
        test_answers_gaussian_mixture_avg_std_SSE_list_lbrack_float_rbrack.explanation = explanation
        assert False
    else:
        student_answer = student_answer['avg_std_SSE']
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
    msg_structure = "assert_utilities.check_structure_list_float(student_answer, instructor_answer)"
    msg_answer = "assert_utilities.check_answer_list_float(student_answer, instructor_answer, rel_tol, monotone_increasing, partial_score_frac)"
    local_namespace.update({'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'keys':keys})
    choices = []
    local_namespace['choices'] = choices
    answer_type = 'list[float]'
    question_id = 'gaussian_mixture'
    subquestion_id = 'avg_std_SSE'
    partial_score_frac_l = [0.]
    local_namespace['partial_score_frac_l'] = partial_score_frac_l
    function_name.answer_type = answer_type
    function_name.question_id = question_id
    function_name.subquestion_id = subquestion_id
    function_name.partial_score_frac = partial_score_frac_l[0]
    is_success, explanation_structure = eval(msg_structure, {'__builtins__':{}}, local_namespace)
    if is_success:
        is_success, explanation_answer    = eval(msg_answer,    {'__builtins__':{}}, local_namespace)
        if is_success is True:
            function_name.partial_score_frac = 1.0
        else:
            function_name.partial_score_frac = partial_score_frac_l[0]
    else: 
        explanation_answer = 'Failed structural tests, No grade for answer component\n.' 
        explanation_answer += f'Instructor answer: {repr(correct_answer)}\n'
        explanation_answer += f'Student answer: {repr(student_answer)}'
        function_name.partial_score_frac = partial_score_frac_l[0]
    explanation = '\n'.join(['==Structure tests==:', explanation_structure, '==Answer tests==:', explanation_answer])
    test_answers_gaussian_mixture_avg_std_SSE_list_lbrack_float_rbrack.explanation = explanation
    assert is_success


