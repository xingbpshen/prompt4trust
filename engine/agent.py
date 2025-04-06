from trl import GRPOConfig, GRPOTrainer
from engine import init_log_path, parse_answer_prob, is_supported_closed_source_model
import os
import dataset
from prompt import build_downstream_prompt
import numpy as np
from openai import OpenAI


class Agent:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.log_path = os.path.join(args.log_folder, args.trial_name)
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

    def reward_func(self, completions, questions, gt_answers, option_lists, **kwargs):
        """
        Reward function for the LLM. The reward function is used to evaluate the quality of the completions.
        :param completions: list of completions from the LLM, it is a list of (list of dicts)s if conversation is used
        :param questions: list of question text
        :param gt_answers: list of ground truth answers for the question
        :param option_lists: list of options (list of strings) for the question
        :param kwargs: other arguments
        :return: The function must return a list of floats. Each float represents the reward corresponding to a single completion.
        """
        # Implement the reward function logic here
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
            text = output.outputs[0].text
            answer, prob = parse_answer_prob(text)
            # use log score
            if answer == gt_answer:
                rewards.append(np.log(prob))
            else:
                rewards.append(np.log(1 - prob))
        return rewards

    def train(self, trainer_name='GRPO'):
        assert trainer_name == 'GRPO'
        # minimum example
        trainer_config = GRPOConfig(output_dir=str(self.log_path),
                                    logging_steps=self.config.train.logging_steps,
                                    temperature=self.config.train.gen_temperature,
                                    top_p=self.config.train.top_p,
                                    top_k=self.config.train.top_k,
                                    use_vllm=self.config.train.use_vllm,
                                    learning_rate=self.config.train.learning_rate,
                                    scale_rewards=self.config.train.scale_rewards,
                                    max_prompt_length=self.config.train.max_prompt_length,
                                    max_completion_length=self.config.train.max_completion_length,
                                    num_generations=self.config.train.num_generations)
        trainer = GRPOTrainer(model=self.config.model.policy,
                              reward_funcs=self.reward_func,
                              args=trainer_config,
                              train_dataset=dataset.get_dataset(args=self.args, config=self.config, split='train'))
        trainer.train(resume_from_checkpoint=False)
