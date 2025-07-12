from datasets import Dataset
import json
import os
from prompt import build_policy_prompt
import csv
from PIL import Image
import matplotlib.pyplot as plt



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

    if config.dataset.name == 'pmcvqa':
        assert split in config.dataset.split_names
        data_path = os.path.join(config.dataset.local_path, f'{split}.csv')  
        # img_dir = os.path.join(config.dataset.local_path, 'images') 

        data_list = []
        with open(data_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            i = 0
            for row in reader:
                question = row['Question'].strip()
                answer_label = row['Answer_label'].strip()
                options = [row['Choice A'].strip(), row['Choice B'].strip(), 
                        row['Choice C'].strip(), row['Choice D'].strip()]
                
                fig_path = row['Figure_path'].strip()
                full_path = os.path.join(config.dataset.image_root, fig_path)
         
                # try:
                #     image = Image.open(full_path).convert("RGB")
                # except Exception as e:
                #     print(f"Warning: could not load image {full_path}. Error: {e}")
                #     continue

                data = {
                    'question': question,
                    'gt_answer': answer_label,
                    'options': options,
                    'image_path': full_path,
                }
        
                conversation = [{
                    "role": "user",
                    "content": build_policy_prompt(
                                dataset_name=config.dataset.name,
                                question_text=question,
                                option_list=options,
                            )
                    # "content": [
                    #     # {
                    #     #     "type": "image",
                    #     #     "image": full_path  
                    #     # },
                    #     {
                    #         # "type": "text",
                    #         "text": build_policy_prompt(
                    #             dataset_name=config.dataset.name,
                    #             question_text=question,
                    #             option_list=options,
                    #         )
                    #     }
                    # ]
                }]

                data['prompt'] = conversation
                data_list.append(data)
        
        return Dataset.from_list(data_list)
    else:
        raise NotImplementedError(f"Dataset {config.dataset.name} not implemented.")
