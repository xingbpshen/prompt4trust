from datasets import Dataset
import json
import os
from prompt import build_policy_prompt


def get_dataset(args, config, split):
    if config.dataset.name == 'medmcqa':
        assert split in config.dataset.split_names
        data_path = os.path.join(config.dataset.local_path, f'{split}.json')
        data_list = []
        with open(data_path, "r") as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    raw_data = json.loads(line) # dict
                    data = {'question': raw_data['question'],
                            'gt_answer': raw_data['cop'],
                            'options': [raw_data[op] for op in ['opa', 'opb', 'opc', 'opd'] if op in raw_data]}
                    conversation = [{'role': 'user', 'content': build_policy_prompt(dataset_name=config.dataset.name,
                                                                                    question_text=data['question'],
                                                                                    option_list=data['options'])}]
                    data['prompt'] = conversation
                    data_list.append(data)
        return Dataset.from_list(data_list)
    else:
        raise NotImplementedError(f"Dataset {config.dataset.name} not implemented.")
