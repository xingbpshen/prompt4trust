import medmcqa


def build_policy_prompt(dataset_name, question_text, option_list):
    if dataset_name == 'medmcqa':
        options_text = "\n".join(f"({i+1}) {text}" for i, text in enumerate(option_list))
        return medmcqa.policy_model_part1 + '\nQuestion:\n' + question_text + '\nOptions:\n' + options_text + '\n\n' + medmcqa.policy_model_part2
    else:
        raise NotImplementedError


def get_match_pattern():
    return r"answer is (\d+(?:\.\d+)?) with confidence (\d+(?:\.\d+)?)"
