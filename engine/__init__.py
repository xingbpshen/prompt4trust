import numpy as np
import os
from prompt import get_match_pattern
import re


def accuracy(gt_answers, lm_answers):
    assert len(gt_answers) == len(lm_answers)
    correct = 0
    for gt, lm in zip(gt_answers, lm_answers):
        # use case-insensitive matching if they are both strings
        if isinstance(gt, str) and isinstance(lm, str):
            if gt.lower() == lm.lower():
                correct += 1
        else:
            if gt == lm:
                correct += 1
    return correct / len(gt_answers)


def ece(gt_answers, lm_answers, lm_probabilities, n_bins=10):
    """
    Expected Calibration Error (ECE) is a scalar that measures the calibration of a model using uniform bins.
    :param gt_answers: list of ground truth answers in character format
    :param lm_answers: list of predicted answers in character format
    :param lm_probabilities: list of predicted probabilities for the lm_answers
    :return: calibration error
    """
    assert len(gt_answers) == len(lm_answers) == len(lm_probabilities)

    gt_answers = np.array(gt_answers)
    lm_answers = np.array(lm_answers)
    lm_probabilities = np.array(lm_probabilities)

    # correctness: 1 if predicted == true else 0
    # convert to lower case if both are strings
    if all(isinstance(x, str) for x in gt_answers) and all(isinstance(x, str) for x in lm_answers):
        gt_answers = np.array([x.lower() for x in gt_answers])
        lm_answers = np.array([x.lower() for x in lm_answers])

    correctness = (lm_answers == gt_answers).astype(float)

    # bin boundaries
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]

        # get indices of predictions in the current bin
        in_bin = (lm_probabilities > bin_lower) & (lm_probabilities <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            avg_confidence = np.mean(lm_probabilities[in_bin])
            avg_accuracy = np.mean(correctness[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence - avg_accuracy)

    return ece


def init_log_path(log_path, args):
    if os.path.exists(log_path) and args.train:
        # ask if user wants to overwrite the existing folder
        response = input(f"Folder {log_path} already exists. Overwrite? (y/n): ")
        # case insensitive check
        response = response.lower()
        if response != "y":
            raise ValueError
        elif response == "y":
            # remove the existing folder and create a new one
            os.system(f"rm -r {log_path}")
            os.makedirs(log_path)
    else:
        return 0


def parse_answer_prob(text):
    match = re.search(r"answer is (\d+(?:\.\d+)?) with confidence (\d+(?:\.\d+)?)", text)
    if match:
        number = float(match.group(1))
        confidence = float(match.group(2))
        return number, confidence
    else:
        # randomly return a number and confidence 0
        return np.random.randint(1, 5), 0


def reward_func(completions, question, gt_answer, options, **kwargs):
    """
    Reward function for the LLM. The reward function is used to evaluate the quality of the completions.
    :param completions: list of completions from the LLM, it is a list of dicts if conversation is used
    :param question: list of question text
    :param gt_answer: list of ground truth answers for the question
    :param options: list of options (list of strings) for the question
    :param kwargs: other arguments
    :return: The function must return a list of floats. Each float represents the reward corresponding to a single completion.
    """
    # Implement the reward function logic here
    # the completions are used for another LLM as prompt
    pass
