policy_model_part1 = "Draft the prompt to assisting another LLM to identify the correct option for a multiple-choice question."

policy_model_part2 = "Prompt draft:"

downstream_model_part1 = """You must provide a NUMBER answer (1, 2, 3, or 4) and a CONFIDENCE score (0-100) in the last sentence of your response as "In conclusion, the answer is {NUMBER} with confidence {CONFIDENCE}" """

downstream_model_part2 = "Hint:"

downstream_model_part3 = "Answer:"
