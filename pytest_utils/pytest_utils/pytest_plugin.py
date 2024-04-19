import pytest
import json

# TODO: Limit scores to 2 significant digits <<<< 


@pytest.mark.hookwrapper
def pytest_runtest_makereport(item):
    x = yield
    x._result.max_score = getattr(item._obj, 'max_score', 0)
    x._result.visibility = getattr(item._obj, 'visibility', 'visible')
    x._result.partial_score_frac = getattr(item._obj, 'partial_score_frac', 1.)
    x._result.explanation = getattr(item._obj, 'explanation', None)
    x._result.answer_note = getattr(item._obj, 'answer_note', None)
    x._result.answer_type = getattr(item._obj, 'answer_type', None)
    x._result.question_id = getattr(item._obj, 'question_id', None)
    x._result.subquestion_id = getattr(item._obj, 'subquestion_id', None)
    x._result.note = getattr(item._obj, 'note', None)
    x._result.hide_errors = getattr(item._obj, 'hide_errors', None)

def pytest_terminal_summary(terminalreporter, exitstatus):
    json_results = {
        'question_id': '',
        'subquestion_id': '',
        'output': 'Text relevant to the entire submission',
        'output_format': 'simple_format',
        'test_output_format': 'text',
        'test_name_format': 'text',
        'stdout_visibility': 'hidden',
        'visibility': 'visible',
        'explanation': '',
        'note': '',
        'answer_note': '',
        'hide_errors': '',
        'partial_score_frac': 1,
        'extra_data': {},
        'tests': []
    }

    all_tests = []
    total_student_score = 0

    if ('failed' in terminalreporter.stats):
        all_tests = all_tests + terminalreporter.stats['failed']
    if ('passed' in terminalreporter.stats):
        all_tests = all_tests + terminalreporter.stats['passed']

    # First, calculate total scores
    total_max_score = 0
    for s in all_tests:
        total_max_score += s.max_score
    if total_max_score > 0:
        global_scaling_factor = 100.  / total_max_score
    else:
        global_scaling_factor = 0.
    #print("GLOBAL FACTOR: ", global_scaling_factor)

    for s in all_tests:
        # Printed after each message
        output = "" 
        rescaled_score = 0
        score = s.max_score

        """
        # Not a good idea to provide links to the test code, because I do not 
        # wish to students to see it. 
        # Construct a clickable link using the file path and line number
        file_path = s.location[0]  # The file path of the test
        line_number = s.location[1]  # The line number of the test
        clickable_link = f"{file_path}:{line_number}"
        # Include the clickable link in the output
        output += f"\nClick here to view the test: {clickable_link}"
        """

        if True:
            if s.question_id is not None:
                output += f"Question: {repr(s.question_id)}\n"

            if s.subquestion_id is not None:
                output += f"Subquestion: {repr(s.subquestion_id)}\n"

            if s.answer_type is not None:
                output += f"Correct answer type: {repr(s.answer_type)}\n"

            if s.explanation is not None:
                # print additional explanation
                output += f"Explanation: {s.explanation!s}\n"

            if s.answer_note is not None and s.answer_note is not "":
                output += f"Additional note: {s.answer_note!s}\n"

            if hasattr(s, 'hide_errors') and s.hide_errors is not None:
                output += f"{s.hide_errors}\n"
            else:
                # print assert error message
                error_message = str(s.longrepr.reprcrash.message)
                output += f"{error_message}\n"

        if (s.outcome == 'failed'):
            score = s.max_score * s.partial_score_frac
            rescaled_score = (score / (total_max_score+.01)) * 100
            status = 'failed'  # not sure if needed
        elif (s.outcome == 'passed'):
            score = s.max_score
            rescaled_score = (s.max_score / (total_max_score+.01)) * 100
            status = 'passed'
        else:
            output = ''
            status = 'failed'

        total_student_score += score 

        score = round(score, 2)
        rescaled_score = round(rescaled_score, 2)
        #print(f"===> {type(s.partial_score_frac)=}")
        s.partial_score_frac = round(s.partial_score_frac, 2)

        output += f"partial_score_frac: {s.partial_score_frac}\n"
        output += f"max_score: {s.max_score}\n"
        output += f"score: {score}\n"
        output += f"rescaled_score: {rescaled_score}\n"

        #print(f"{score=}")
        #print(f"{total_max_score=}")
        #print(f"{s.partial_score_frac=}")
        #print(f"{global_scaling_factor=}")
        #print(f"{s.max_score=}")
        #print(f"'cor': {round(score * global_scaling_factor, 2)=}")

        json_results["tests"].append(
            {
                'question_id': s.question_id,
                'subquestion_id': s.subquestion_id,
                #'score': rescaled_score,
                'score': round(score * global_scaling_factor, 2),
                'partial_score_frac': s.partial_score_frac,
                #'partial_score': score,
                'max_score': round(s.max_score * global_scaling_factor, 2),
                # 'max_score': s.max_score,
                'answer_type': s.answer_type,
                'name': s.location[2],
                'output': output,
                'visibility': s.visibility,
                'explanation': s.explanation,
                'status': status
            }
        )

    with open('results.json', 'w') as results:
        #print(json_results)
        results.write(json.dumps(json_results, indent=4))

