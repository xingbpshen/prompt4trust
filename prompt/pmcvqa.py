policy_model_part1 = "Draft a prompt to help a vision-language model answer a multiple-choice question about an image with calibrated confidence."

policy_model_part2 = "Prompt draft:"

downstream_model_part1 = """Clearly state your final answer and confidence in the following: "In conclusion, the answer is {LETTER} with confidence {CONFIDENCE}". Replace {LETTER} with one of A, B, C, or D, and {CONFIDENCE} with your confidence in your answer between 0 and 100.After giving the answer, explain your reasoning based on the image and the question. """

downstream_model_part2 = "Hint: "

downstream_model_part3 = "Answer:"