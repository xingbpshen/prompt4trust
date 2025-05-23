policy_model_part1 =  """
                        "You are assisting a downstream vision-language model (VLM) with a multiple-choice question about an image. "
                        "You do not have access to the image or the correct answer, Do not give an answer to the question. "
                        "Your task is to write a neutral hint based only on the question and answer options, "
                        "to guide the VLM on what visual evidence to consider when answering. "
                        "The VLM will use the question, your hint, and the image to give a response that also reflects its own confidence."
                    """


policy_model_part2 = "Prompt draft:"

downstream_model_part1 = """"You are answering a multiple-choice question with four options (A, B, C, or D). "
                        "Clearly state your final answer and confidence in the following format only:\n\n"
                        "'In conclusion, the answer is {LETTER} with confidence {CONFIDENCE}.'\n\n"
                        "Replace {LETTER} with one of A, B, C, or D, and {CONFIDENCE} with a number between 0 and 100.\n"
                        "After giving the answer, explain your reasoning based on the image and the question." """

downstream_model_part2 = "Hint: "

downstream_model_part3 = "Answer:"