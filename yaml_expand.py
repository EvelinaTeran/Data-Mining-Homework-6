"""
Expand a yaml file that contains answer lists so that there is only a single answer per question.
"""

import yaml
import pytest
from generator_utils import sanitize_function_name
import argparse
from copy import deepcopy


def del_var(var, key):
    try:
        del var[key]
    except (NameError, KeyError):
        # print("error del_var")
        pass


def load_yaml_file(file_path):
    """
    Load a YAML file and return its contents as a dictionary.

    Args:
        file_path (str): The path to the YAML file.

    Returns:
        dict: The contents of the YAML file as a dictionary.
    """
    with open(file_path, "r") as file:
        questions_data = yaml.safe_load(file)
    return questions_data


def main(f_in: str, f_out: str):
    """
    Main function to expand a YAML file.

    Args:
        f_in (str): The path to the input YAML file.
        f_out (str): The path to the output YAML file.
    """
    # Read the yaml file f_in
    questions_data = load_yaml_file(f_in)
    expanded_questions_data = deepcopy(questions_data)

    for question in expanded_questions_data["questions"]:
        #print("question: ", question)
        new_parts = []
        parts = question["parts"]
        for part in parts:
            #print(f"{part=}")
            s_answer = [part.get("s_answer", None)]
            i_answer = [part.get("i_answer", None)]
            #print(f"{s_answer=}")
            #print(f"{part.get('s_answers', [None])=}")
            s_answers = part.get("s_answers", [None]) + s_answer
            i_answers = part.get("i_answers", [None]) + i_answer
            # print(f"{s_answers=}")
            # print(f"{i_answers=}")
            # print(f"{s_answer=}")
            # print(f"{i_answer=}")
            i_answers = [i for i in i_answers if i is not None and i is not [None]]
            s_answers = [i for i in s_answers if i is not None and i is not [None]]
            #print(f"=> {s_answers=}")
            #print(f"=> {i_answers=}")
            # works if either s_answer or s_answers is present
            """
            if s_answers is not None and not isinstance(s_answers, list):
                print("s_answers must be absent or a list")
                raise "Error: s_answers wrong type"
            if i_answers is not None and not isinstance(i_answers, list):
                print("i_answers must be absent or a list")
                raise "Error: i_answers wrong type"
            if s_answers is None and s_answer is not None:
                s_answers = list(s_answer)
            if i_answers is None and i_answer is not None:
                print("i_answer: ", i_answer)
                i_answers = list(i_answer)
            """

            s_answers = [None] if s_answers == [] else s_answers
            i_answers = [None] if i_answers == [] else i_answers

            for s, s_answer in enumerate(s_answers):
                for i, i_answer in enumerate(i_answers):
                    new_part = deepcopy(part)
                    del_var(new_part, "s_answers")
                    del_var(new_part, "i_answers")
                    if i_answer is not None and s_answer is not None:
                        new_part["id"] += f"_{i}_{s}"
                    # print("==> answer= ", answer)
                    if s_answer is not None:
                        new_part["s_answer"] = s_answer
                    if i_answer is not None:
                        new_part["i_answer"] = i_answer
                    new_parts.append(new_part)
                    #print("new_part: ", new_part)
        question["parts"] = new_parts

    with open(f_out, "w") as outfile:
        yaml.safe_dump(expanded_questions_data, outfile, default_flow_style=False)


# ----------------------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Pass in the name of the input yaml file."
    )
    parser.add_argument("-y", "--yaml", help="Name of the yaml file", required=True)
    parser.add_argument(
        "-o", "--out", help="Name of the output yaml file", required=True
    )
    args = parser.parse_args()
    main(args.yaml, args.out)
