import os
import re
import time

import numpy as np
import requests
import util
from openai import OpenAI
from prompt import (get_answer_match_pattern, get_confidence_match_pattern,
                    get_match_pattern)

INVALID_RESPONSE_FORMAT_PENALTY = 1e-12


def compute_accuracy(gt_answers, lm_answers):
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


def compute_ece(gt_answers, lm_answers, lm_probabilities, n_bins=10):
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

        if bin_lower == 0.0:  # EDGE CASE for 0 prob
            in_bin = (lm_probabilities >= bin_lower) & (lm_probabilities <= bin_upper)
        else:
            in_bin = (lm_probabilities > bin_lower) & (lm_probabilities <= bin_upper)

        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            avg_confidence = np.mean(lm_probabilities[in_bin])
            avg_accuracy = np.mean(correctness[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence - avg_accuracy)

    return ece


def compute_brier_score(gt_answers, lm_answers, lm_probabilities):
    """
    Brier score: measure of accuracy of confidence predictions
    :param gt_answers: list of ground truth answers in character format
    :param lm_answers: list of predicted answers in character format
    :param lm_probabilities: list of predicted probabilities for the lm_answers
    :return: brier score
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

    # calculate brier score
    brier_score = np.mean((lm_probabilities-correctness) ** 2)

    return brier_score


# changed to be able to use checkpoints in the given log path
def init_log_path(log_path, args):
    if os.path.exists(log_path):
        if args.train and not args.resume:
            if not args.ni:
                response = input(
                    f"Folder {log_path} already exists. Overwrite? (y/n): ").lower()
                if response != "y":
                    raise ValueError(
                        f"Folder {log_path} already exists. Please remove it or choose a different folder.")
            os.system(f"rm -r {log_path}")
            os.makedirs(log_path)
        elif args.train and args.resume:
            print(
                f"Resuming checkpoints in existing folder {log_path}, not overwriting.")
        else:
            os.makedirs(log_path, exist_ok=True)
    else:
        os.makedirs(log_path)


def parse_answer_prob_vqa(text):
    """Extracts predicted answer letter and confidence score from model output."""
    answer_match = re.search(r"answer is\s+([A-D])", text, re.IGNORECASE)
    confidence_match = re.search(
        r"confidence\s+(\d{1,3})", text, re.IGNORECASE)
    if answer_match and confidence_match:
        pred = answer_match.group(1).upper()
        confidence = min(float(confidence_match.group(1)), 100.0) / 100
        return pred, confidence
    return -1, INVALID_RESPONSE_FORMAT_PENALTY


def parse_answer_prob(text):
    # match = re.search(get_match_pattern(), text)
    # Use separate confidence and answer match patterns to accept responses with slight format deviations
    answer_match = re.search(get_answer_match_pattern(), text)
    confidence_match = re.search(get_confidence_match_pattern(), text)
    # print(f'ANSWER MATCH: {answer_match}, CONFIDENCE_MATCH {confidence_match}')
    if answer_match and confidence_match:
        try:
            number = int(answer_match.group(1))
        except ValueError:
            # Handle edge cases where answer is not reported as an int
            number = str(answer_match.group(1))
        # Convert percentage to confidence score
        confidence = float(confidence_match.group(1))/100
        return number, confidence
    else:
        # return an invalid answer number and confidence nearly 0
        # INVALID_RESPONSE_FORMAT_PENALTY defines the strength of the penalty
        # for responses that don't report confidence in a way that we can parse
        # print('Did not find answer and confidence score')
        return -1, INVALID_RESPONSE_FORMAT_PENALTY


def is_ready(port):
    """
    Check if the vllm server is ready to accept requests.
    :param port: The port on which the vllm server is running.
    :return: True if the server is ready, False otherwise.
    """
    url = f"http://localhost:{port}/health/"
    try:
        response = requests.get(url)
    except requests.exceptions.RequestException as exc:
        return False
    else:
        if response.status_code == 200:
            return True


def wait_until_ready(port, subproc, timeout=1800):
    """
    Wait until the vllm server is ready to accept requests.
    :param port: The port on which the vllm server is running.
    :param subproc: The subprocess object for the vllm server.
    :param timeout: The maximum time to wait in seconds.
    """
    start_time = time.time()
    while not is_ready(port):
        # if the server has exited, raise an error
        if subproc.poll() is not None:
            stderr_output = subproc.stderr.read().decode()
            raise RuntimeError(
                f"Error:\n{stderr_output}\nvLLM server at port {port} exited unexpectedly, please kill the corresponding GPU process manually by:\nkill -9 PID")
        if time.time() - start_time > 30:
            util.info('engine.__init__.py',
                      'Still waiting? Check the GPU mem usage to make sure no server is lost.')
        if time.time() - start_time > timeout:
            raise TimeoutError(
                f"Server at port {port} did not become ready within {timeout} seconds.")
        time.sleep(10)


def is_supported_closed_source_model(model_name):
    """
    Returns the provider name and base_url of the closed source model if it is supported.
    """
    if model_name in ['gemini-2.0-flash-001']:
        return 'google', 'https://generativelanguage.googleapis.com/v1beta/openai/'
    elif model_name in ['gpt-4o-mini-2024-07-18']:
        return 'openai', None
    else:
        raise ValueError(f"Closed-source model {model_name} is not supported.")


def convert_letter_to_idx(letter: str) -> int:
    if type(letter) is int and letter == -1: 
        return -1
    letter = letter.upper()
    if letter == 'A':
        return 1
    elif letter == 'B':
        return 2
    elif letter == 'C':
        return 3
    elif letter == 'D':
        return 4
    else:
        raise ValueError(
            f"Invalid letter: {letter}. Expected one of A, B, C, D.")
