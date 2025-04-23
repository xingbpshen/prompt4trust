from trl import GRPOConfig, GRPOTrainer
from engine import init_log_path, parse_answer_prob, is_supported_closed_source_model, INVALID_RESPONSE_FORMAT_PENALTY, compute_accuracy, compute_ece
import os
import dataset
from prompt import build_downstream_prompt
import numpy as np
from openai import OpenAI
import torch
from transformers.trainer_utils import get_last_checkpoint
from pprint import pprint
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json

class Agent:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.log_path = os.path.join(args.log_folder, args.trial_name)
        if os.path.exists(self.log_path):
            checkpoint_path = get_last_checkpoint(self.log_path) #check if there is a saved checkpoint
        else: 
            checkpoint_path = None
        if checkpoint_path: #if it exists, resume from the checkpoint
            self.checkpoint_path = checkpoint_path
            self.args.resume = True
        else: #otherwise, start from scratch
            self.checkpoint_path = None
            self.args.resume = False
        init_log_path(self.log_path, args)
        if args.is_closed_source_downstream:
            provider, base_url = is_supported_closed_source_model(config.model.downstream)
            api_key = getattr(config.api_key, provider)
        else:
            api_key = 'EMPTY'
            base_url = f'http://localhost:{config.resources.downstream_port}/v1'
        self.downstream_model = OpenAI(api_key=api_key,
                                       base_url=base_url)

    def send_message_downstream(self, message):
        """
        Send a message to the downstream model and get the response.
        :param message: a dict of conversation, e.g. {'role': 'user', 'content': 'How to cook chicken curry?'}
        :return: the response from the downstream model, pure text
        """
        if self.args.is_closed_source_downstream:
            model = self.config.model.downstream
        else:
            models = self.downstream_model.models.list()
            model = models.data[0].id
        chat_completion = self.downstream_model.chat.completions.create(model=model,
                                                                        temperature=self.config.downstream.gen_temperature,
                                                                        top_p=self.config.downstream.top_p,
                                                                        max_completion_tokens=self.config.downstream.max_completion_tokens,
                                                                        n=1,
                                                                        messages=[message])
        return chat_completion.choices[0].message.content

    def reward_func(self, completions, **kwargs):
        """
        Reward function for the LLM. The reward function is used to evaluate the quality of the completions.
        :param completions: list of completions from the LLM, it is a list of (list of dicts)s if conversation is used
        :param kwargs: other arguments
        :return: The function must return a list of floats. Each float represents the reward corresponding to a single completion.
        """
        # Implement the reward function logic here
        questions = kwargs.get('question', None)
        gt_answers = kwargs.get('gt_answer', None)
        option_lists = kwargs.get('options', None)

        assert questions is not None and gt_answers is not None and option_lists is not None
        # the completions are used for another LLM as prompt
        conversation_list = []
        for completion, question, option_list in zip(completions, questions, option_lists):
            # build the prompt
            prompt = build_downstream_prompt(dataset_name=self.config.dataset.name,
                                             question_text=question,
                                             option_list=option_list,
                                             hint_text=completion[0]['content'])
            conversation_list.append({'role': 'user', 'content': prompt})
        # run the downstream model
        outputs = []
        for conversation in conversation_list:
            outputs.append(self.send_message_downstream(conversation))
        # get the answer and prob from the outputs
        rewards = []
        for output, gt_answer in zip(outputs, gt_answers):
            text = output
            answer, prob = parse_answer_prob(text)
            # use log score
            if answer == -1 and prob == INVALID_RESPONSE_FORMAT_PENALTY: 
                # Response did not include confidence report or was formatted 
                # such that the reported confidence could not be parsed
                rewards.append(np.log(prob))
  
            elif answer == gt_answer:
                # clip prob to avoid log(0)
                prob = min(1, max(prob, 1e-10))
                rewards.append(np.log(prob))

            else:
                tmp = min(1, max(1 - prob, 1e-10))
                rewards.append(np.log(tmp))
                
        return rewards

    def train(self, trainer_name='GRPO'):
        assert trainer_name == 'GRPO'
        # minimum example
        trainer_config = GRPOConfig(output_dir=str(self.log_path),
                                    logging_steps=self.config.train.logging_steps,
                                    save_steps = self.config.train.save_steps, #checkpointing at 250 steps
                                    temperature=self.config.train.gen_temperature,
                                    top_p=self.config.train.top_p,
                                    top_k=self.config.train.top_k,
                                    use_vllm=self.config.train.use_vllm,
                                    vllm_server_host='localhost',
                                    vllm_server_port=self.config.resources.action_port,
                                    learning_rate=float(self.config.train.learning_rate),
                                    scale_rewards=self.config.train.scale_rewards,
                                    max_prompt_length=self.config.train.max_prompt_length,
                                    max_completion_length=self.config.train.max_completion_length,
                                    num_generations=self.config.train.num_generations,
                                    save_total_limit=3, 
                                    max_grad_norm=self.config.train.max_grad_norm, 
                                    num_iterations=self.config.train.num_iterations, 
                                    per_device_train_batch_size=self.config.train.per_device_train_batch_size, 
                                    beta = self.config.train.beta
                                ) 
                                    
        trainer = GRPOTrainer(model=self.config.model.policy,
                              reward_funcs=self.reward_func,
                              args=trainer_config,
                              train_dataset=dataset.get_dataset(args=self.args, config=self.config, split=self.config.dataset.split_names[0]))

        trainer.train(resume_from_checkpoint=self.checkpoint_path)

    def eval(self):

        # 1. Load model and tokenizer from checkpoint
        print(f"Loading model from checkpoint: {self.checkpoint_path}")
        model = AutoModelForCausalLM.from_pretrained(self.checkpoint_path).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path)
        model.eval()

        # 2. Load dataset
        eval_dataset = dataset.get_dataset(
            args=self.args,
            config=self.config,
            # TODO change this hard coding
            split=self.config.dataset.split_names[2]
        )

        # 3. Loop through examples and generate completions, then send to downstream model 
        calibrated_lm_answers = []
        calibrated_lm_probabilities = []
        baseline_lm_answers = []
        baseline_lm_probabilities = []
        for sample in tqdm(eval_dataset):
            question = sample["question"]
            options = sample["options"]
            gt_answer = sample["gt_answer"]
            prompt = sample["prompt"]
        
            inputs = tokenizer(prompt[0]["content"], return_tensors="pt").to("cuda")
      
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=self.config.train.max_completion_length,
                    temperature=self.config.train.gen_temperature,
                    top_p=self.config.train.top_p,
                    do_sample=True
                )
            completion = tokenizer.decode(output[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
         
            calibrated_prompt = build_downstream_prompt(
                dataset_name=self.config.dataset.name,
                question_text= question,
                option_list = options,
                hint_text=completion,
            )

            calibrated_output = self.send_message_downstream({'role': 'user', 'content': calibrated_prompt})
            calibrated_answer, calibrated_prob = parse_answer_prob(calibrated_output)

            calibrated_lm_answers.append(calibrated_answer)
            calibrated_lm_probabilities.append(calibrated_prob)

            baseline_prompt = build_downstream_prompt(
                dataset_name=self.config.dataset.name,
                question_text=question,
                option_list=options,
                hint_text= None,
            )

            baseline_output = self.send_message_downstream({'role': 'user', 'content': baseline_prompt})
            baseline_answer, baseline_prob = parse_answer_prob(baseline_output)

            baseline_lm_answers.append(baseline_answer)
            baseline_lm_probabilities.append(baseline_prob)
        
        # 4. Calcute metrics
        calibrated_acc = compute_accuracy(eval_dataset["gt_answer"], calibrated_lm_answers)
        calibrated_ece = compute_ece(eval_dataset["gt_answer"], calibrated_lm_answers, calibrated_lm_probabilities)

        baseline_acc = compute_accuracy(eval_dataset["gt_answer"], baseline_lm_answers)
        baseline_ece = compute_ece(eval_dataset["gt_answer"], baseline_lm_answers, baseline_lm_probabilities)

        # 5. Log Metrics
        results = {
            "calibrated_accuracy": calibrated_acc,
            "calibrated_ece": calibrated_ece,
            "baseline_accuracy": baseline_acc,
            "baseline_ece": baseline_ece,
            "checkpoint": self.checkpoint_path,
        }

        log_file = os.path.join(self.log_path, "eval_results.json")
        os.makedirs(self.log_path, exist_ok=True)

        with open(log_file, "w") as f:
            json.dump(results, f, indent=4)

        print(f"Metric Results saved to {log_file}")



