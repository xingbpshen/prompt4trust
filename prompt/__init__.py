from prompt import medmcqa, pmcvqa


def build_policy_prompt(dataset_name, question_text, option_list, image = None):
    if dataset_name == 'medmcqa':
        options_text = "\n".join(f"({i+1}) {text}" for i, text in enumerate(option_list))
        return medmcqa.policy_model_part1 + '\nQuestion:\n' + question_text + '\nOptions:\n' + options_text + '\n\n' + medmcqa.policy_model_part2
    if dataset_name == 'pmcvqa':
        options_text = "\n".join(f"({i+1}) {text}" for i, text in enumerate(option_list))
        return (
            pmcvqa.policy_model_part1
            + '\nQuestion:\n' + question_text
            + '\nOptions:\n' + options_text
            + '\n\n' + pmcvqa.policy_model_part2
        )
    else:
        raise NotImplementedError


def build_downstream_prompt(dataset_name, question_text, option_list, hint_text):
    if dataset_name == 'medmcqa':
        options_text = "\n".join(f"({i+1}) {text}" for i, text in enumerate(option_list))
        if hint_text is None: #i.e. baseline
            return medmcqa.downstream_model_part1 + '\nQuestion:\n' + question_text + '\nOptions:\n' + options_text + '\n\n' + medmcqa.downstream_model_part3
        else:
            return medmcqa.downstream_model_part1 + '\nQuestion:\n' + question_text + '\nOptions:\n' + options_text + '\n' + medmcqa.downstream_model_part2 + hint_text + '\n\n' + medmcqa.downstream_model_part3
    if dataset_name == 'pmcvqa':
        options_text = "\n".join(f"({i+1}) {text}" for i, text in enumerate(option_list))
        if hint_text is None: #i.e. baseline
            return pmcvqa.downstream_model_part1 + '\nQuestion:\n' + question_text + '\nOptions:\n' + options_text + '\n\n' + pmcvqa.downstream_model_part3
        else:
            return pmcvqa.downstream_model_part1 + '\nQuestion:\n' + question_text + '\nOptions:\n' + options_text + '\n' + pmcvqa.downstream_model_part2 + hint_text + '\n\n' + pmcvqa.downstream_model_part3
    else:
        raise NotImplementedError


def get_match_pattern():
    return r"answer is (\d+(?:\.\d+)?) with confidence (\d+(?:\.\d+)?)"

def get_answer_match_pattern():
    return r"answer is \(?(\d+(?:\.\d+)?)\)?"

def get_confidence_match_pattern():
    return r"confidence \(?(\d+(?:\.\d+)?)\)?"
