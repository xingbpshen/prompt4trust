import csv
import json
import os
import dataset
import numpy as np
import pandas as pd
import torch
from openai import OpenAI
from prompt import build_downstream_prompt
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoProcessor, AutoTokenizer,
                          Qwen2VLForConditionalGeneration, set_seed)
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer
from engine import (INVALID_RESPONSE_FORMAT_PENALTY, compute_accuracy,
                    compute_brier_score, compute_ece, convert_letter_to_idx,
                    init_log_path, is_supported_closed_source_model,
                    parse_answer_prob, parse_answer_prob_vqa)

set_seed(1)


def filter_valid_samples(gt_answers, lm_answers, lm_probabilities):
    """
    Returns a boolean mask for valid samples:
    - answer not -1
    - confidence not 1e-12
    """
    valid_mask = []
    for gt, lm, conf in zip(gt_answers, lm_answers, lm_probabilities):
        is_valid_answer = (lm != -1) and (conf != 1e-12)
        valid_mask.append(is_valid_answer)
    return np.array(valid_mask)


class Agent:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.log_path = os.path.join(args.log_folder, args.trial_name)
        if os.path.exists(self.log_path):
            # check if there is a saved checkpoint
            checkpoint_path = get_last_checkpoint(self.log_path)
        else:
            checkpoint_path = None
        if checkpoint_path:  # if it exists, resume from the checkpoint
            self.checkpoint_path = checkpoint_path
            self.args.resume = True
        else:  # otherwise, start from scratch
            self.checkpoint_path = None
            self.args.resume = False
        init_log_path(self.log_path, args)
        if args.is_closed_source_downstream:
            provider, base_url = is_supported_closed_source_model(
                config.model.downstream)
            api_key = getattr(config.api_key, provider)
        else:
            api_key = 'EMPTY'
            base_url = f'http://localhost:{config.resources.downstream_port}/v1'

        self.downstream_model = OpenAI(api_key=api_key,
                                       base_url=base_url,
                                       timeout=30.0,
                                       max_retries=0)

    def send_message_downstream(self, message, seed=None):
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
        try:
            chat_completion = self.downstream_model.chat.completions.create(
                model=model,
                temperature=self.config.downstream.gen_temperature,
                top_p=self.config.downstream.top_p,
                max_completion_tokens=self.config.downstream.max_completion_tokens,
                n=1,
                messages=[message],
                seed=seed
            )
            return chat_completion.choices[0].message.content

        except Exception as e:
            print("Error:", e)
            return "Invalid response"

    # for using in eval
    # for greedy, temperature=0.0, top_p=1.0
    def send_message_openai_obj(self, openai_obj, message, temperature, top_p, max_completion_tokens, n=1, seed=None):
        models = openai_obj.models.list()
        model = models.data[0].id
        chat_completion = openai_obj.chat.completions.create(
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_completion_tokens=max_completion_tokens,
            n=n,
            messages=[message],
            seed=seed
        )
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
        image_paths = kwargs.get('image_path', None)

        assert questions is not None and gt_answers is not None and option_lists is not None and image_paths is not None
        # the completions are used for another LLM as prompt
        conversation_list = []
        for completion, question, option_list, image_path in zip(completions, questions, option_lists, image_paths):
            # build the prompt
            prompt = build_downstream_prompt(
                dataset_name=self.config.dataset.name,
                question_text=question,
                option_list=option_list,
                hint_text=completion[0]['content']
            )
            if self.config.dataset.name == 'medmcqa':
                conversation_list.append({'role': 'user', 'content': prompt})
            elif self.config.dataset.name == 'pmcvqa':
                absolute_image_path = os.path.abspath(image_path)
                assert absolute_image_path.startswith(
                    self.config.dataset.image_root)
                assert os.path.exists(absolute_image_path)
                image_url = f"file://{absolute_image_path}"
                conversation_list.append(
                    {
                        'role': 'user',
                        'content': [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url,
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            }
                        ]

                    }
                )

        # run the downstream model
        outputs = []
        for conversation in conversation_list:
            outputs.append(self.send_message_downstream(conversation))
        # get the answer and prob from the outputs
        rewards = []
        probs = []
        predictions = []
        for output, gt_answer in zip(outputs, gt_answers):
            text = output
            if self.config.dataset.name == 'pmcvqa':
                answer, prob = parse_answer_prob_vqa(text)

            elif self.config.dataset.name == 'medmcqa':
                answer, prob = parse_answer_prob(text)

            probs.append(prob)
            predictions.append(answer)

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
                rewards.append(np.log(tmp) - 1)

        # Compute additional metrics for logging purposes
        # modified such that each batch has more than one question
        batch_log = []
        acc = compute_accuracy(gt_answers, predictions)
        ece = compute_ece(gt_answers, predictions, probs)
        brier = compute_brier_score(gt_answers, predictions, probs)

        batch_log.append({
            "accuracy": acc,
            "ece": ece,
            "brier": brier,
            "gt_answers": gt_answers,
            "predictions": predictions,
            "probs": probs,
            "rewards": rewards
        })

        batch_df = pd.DataFrame(batch_log)
        # Open csv file and append the batch log
        if not os.path.exists(os.path.join(self.log_path, "train_log.csv")):
            batch_df.to_csv(os.path.join(
                self.log_path, "train_log.csv"), mode='w', header=True, index=False)
        else:
            batch_df.to_csv(os.path.join(
                self.log_path, "train_log.csv"), mode='a', header=False, index=False)

        return rewards

    def train(self, trainer_name='GRPO'):
        assert trainer_name == 'GRPO'
        # minimum example
        trainer_config = GRPOConfig(
            output_dir=str(self.log_path),
            logging_steps=self.config.train.logging_steps,
            save_steps=self.config.train.save_steps,
            temperature=self.config.train.gen_temperature,
            top_p=self.config.train.top_p,
            top_k=self.config.train.top_k,
            use_vllm=self.config.train.use_vllm,
            vllm_server_host='localhost',
            vllm_server_port=self.config.resources.action_port,
            learning_rate=float(
                self.config.train.learning_rate),
            scale_rewards=self.config.train.scale_rewards,
            max_prompt_length=self.config.train.max_prompt_length,
            max_completion_length=self.config.train.max_completion_length,
            num_generations=self.config.train.num_generations,
            save_total_limit=3,
            max_grad_norm=self.config.train.max_grad_norm,
            num_iterations=self.config.train.num_iterations,
            per_device_train_batch_size=self.config.train.per_device_train_batch_size,
            beta=self.config.train.beta
        )

        trainer = GRPOTrainer(
            model=self.config.model.policy,
            reward_funcs=self.reward_func,
            args=trainer_config,
            train_dataset=dataset.get_dataset(
                args=self.args, config=self.config, split=self.config.dataset.split_names[0])
        )

        trainer.train(resume_from_checkpoint=self.checkpoint_path)

    def eval(self):
        policy_openai_obj = OpenAI(api_key='EMPTY', base_url=f'http://localhost:{self.config.resources.action_port}/v1')
        eval_dataset = dataset.get_dataset(
            args=self.args,
            config=self.config,
            split=self.config.dataset.split_names[2]
        )
        # Create a CSV to hold data at each iteration and track samples that have already been evaluated
        completed_indices = set()
        os.makedirs(self.log_path, exist_ok=True)
        csv_path = os.path.join(self.log_path, "results_log.csv")
        log_file = os.path.join(self.log_path, "results.json")

        if os.path.exists(csv_path) and os.stat(csv_path).st_size > 0:
            print(f"Reading existing CSV at: {csv_path}")
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        completed_indices.add(int(row["index"]))
                    except ValueError:
                        continue  # skip bad rows

        print(f"Resuming from {len(completed_indices)} completed entries")

        fieldnames = [
            "index", "image_path", "question", "gt_answer",
            "calibrated_answer", "calibrated_prob", "calibrated_output",
            "baseline_answer", "baseline_prob", "baseline_output"
        ]

        if self.args.entropy:
            fieldnames.append("calibrated_entropy")
            fieldnames.append("baseline_entropy")

        # Use `with` to guarantee flush and close even if crash occurs
        with open(csv_path, "a", newline="") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            # Write header only if empty file
            if os.stat(csv_path).st_size == 0:
                csv_writer.writeheader()
            # 3. Loop through examples and generate completions, then send to downstream model
            calibrated_lm_answers = []
            calibrated_lm_probabilities = []
            calibrated_entropies = []
            baseline_lm_answers = []
            baseline_lm_probabilities = []
            baseline_entropies = []
            for i, sample in enumerate(tqdm(eval_dataset)):
                if i in completed_indices:
                    continue

                try:
                    question = sample["question"]
                    options = sample["options"]
                    gt_answer = sample["gt_answer"]
                    conversation = sample["prompt"]

                    if self.config.dataset.name == 'pmcvqa':
                        image_path = sample["image_path"]
                        absolute_image_path = os.path.abspath(image_path)
                        assert absolute_image_path.startswith(
                            self.config.dataset.image_root)
                        assert os.path.exists(absolute_image_path)
                        image_url = f"file://{absolute_image_path}"

                    # use greedy search here
                    completion = self.send_message_openai_obj(openai_obj=policy_openai_obj,
                                                              message=conversation[0],
                                                              temperature=0.0,
                                                              top_p=1.0,
                                                              max_completion_tokens=self.config.train.max_completion_length,
                                                              n=1,
                                                              seed=1)

                    calibrated_prompt = build_downstream_prompt(
                        dataset_name=self.config.dataset.name,
                        question_text=question,
                        option_list=options,
                        hint_text=completion,
                    )

                    if self.config.dataset.name == 'medmcqa':
                        calibrated_output = self.send_message_downstream(
                            {'role': 'user', 'content': calibrated_prompt}, seed=1)
                        calibrated_answer, calibrated_prob = parse_answer_prob(
                            calibrated_output)

                    elif self.config.dataset.name == 'pmcvqa':
                        calibrated_output = self.send_message_downstream(
                            {
                                'role': 'user',
                                'content': [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": image_url,
                                        }
                                    },
                                    {
                                        "type": "text",
                                        "text": calibrated_prompt,
                                    }
                                ]
                            },
                            seed=1
                        )

                        calibrated_answer, calibrated_prob = parse_answer_prob_vqa(
                            calibrated_output)

                    else:
                        raise ValueError(
                            'Invalid dataset. Only medmcqa and pmcvqa are supported.')

                    calibrated_lm_answers.append(calibrated_answer)
                    calibrated_lm_probabilities.append(calibrated_prob)

                    baseline_prompt = build_downstream_prompt(
                        dataset_name=self.config.dataset.name,
                        question_text=question,
                        option_list=options,
                        hint_text=None,
                    )

                    if self.config.dataset.name == 'medmcqa':
                        baseline_output = self.send_message_downstream(
                            {'role': 'user', 'content': baseline_prompt}, seed=1)
                        baseline_answer, baseline_prob = parse_answer_prob(
                            baseline_output)

                    elif self.config.dataset.name == 'pmcvqa':
                        baseline_output = self.send_message_downstream(
                            {
                                'role': 'user',
                                'content': [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": image_url,
                                        }
                                    },
                                    {
                                        "type": "text",
                                        "text": baseline_prompt,
                                    }
                                ]
                            },
                            seed=1
                        )

                        baseline_answer, baseline_prob = parse_answer_prob_vqa(
                            baseline_output)


                    else:
                        raise ValueError(
                            'Invalid dataset. Only medmcqa and pmcvqa are supported.')

                    baseline_lm_answers.append(baseline_answer)
                    baseline_lm_probabilities.append(baseline_prob)

                    if self.args.entropy:
                        # Implement uncertainty quantification, referencing: [1] Q.
                        # Lyu et al., “Calibrating Large Language Models with Sample Consistency”.
                        # Use default of 40 MC samples (consistent with Lyu et al.)

                        mc_samples = 40  # TODO: Make this configurable

                        opt_count = len(options)

                        cal_opt_freq = np.zeros((opt_count))
                        base_opt_freq = np.zeros((opt_count))
                        for mc_sample in tqdm(range(mc_samples)):

                            if self.config.dataset.name == 'medmcqa':
                                ent_calibrated_output = self.send_message_downstream(
                                    {'role': 'user', 'content': calibrated_prompt})
                                ent_baseline_output = self.send_message_downstream(
                                    {'role': 'user', 'content': baseline_prompt})
                                ent_calibrated_answer, ent_calibrated_prob = parse_answer_prob(
                                    ent_calibrated_output)
                                ent_baseline_answer, ent_baseline_prob = parse_answer_prob(
                                    ent_baseline_output)

                            elif self.config.dataset.name == 'pmcvqa':
                                ent_calibrated_output = self.send_message_downstream(
                                    {
                                        'role': 'user',
                                        'content': [
                                            {
                                                "type": "image_url",
                                                "image_url": {
                                                    "url": image_url,
                                                }
                                            },
                                            {
                                                "type": "text",
                                                "text": calibrated_prompt,
                                            }
                                        ]
                                    }
                                )

                                ent_baseline_output = self.send_message_downstream(
                                    {
                                        'role': 'user',
                                        'content': [
                                            {
                                                "type": "image_url",
                                                "image_url": {
                                                    "url": image_url,
                                                }
                                            },
                                            {
                                                "type": "text",
                                                "text": baseline_prompt,
                                            }
                                        ]
                                    }
                                )

                                ent_calibrated_answer, ent_calibrated_prob = parse_answer_prob_vqa(
                                    ent_calibrated_output)
                                ent_baseline_answer, ent_baseline_prob = parse_answer_prob_vqa(
                                    ent_baseline_output)

                                ent_calibrated_answer = convert_letter_to_idx(ent_calibrated_answer)
                                ent_baseline_answer = convert_letter_to_idx(ent_baseline_answer)

                            else:
                                raise ValueError('Invalid dataset. Only MedMCQA and PMCVQA are supported.')

                            # Don't count invalid responses (-1 or non-int format)
                            if type(ent_calibrated_answer) == int and ent_calibrated_answer >= 0 and ent_calibrated_answer - 1 < opt_count:
                                # Subtract 1 for indexing since multiple choice answer numbers are not zero-indexed
                                cal_opt_freq[ent_calibrated_answer - 1] += 1

                            # Don't count invalid responses (-1 or non-int format)
                            if type(ent_baseline_answer) == int and ent_baseline_answer >= 0 and ent_baseline_answer - 1 < opt_count:
                                # Subtract 1 for indexing since multiple choice answer numbers are not zero-indexed
                                base_opt_freq[ent_baseline_answer - 1] += 1

                        # Normalize frequencies based on number of valid answers
                        cal_opt_freq = cal_opt_freq / np.sum(cal_opt_freq)
                        base_opt_freq = base_opt_freq / np.sum(base_opt_freq)

                        calibrated_option_entropy = np.zeros((opt_count))
                        baseline_option_entropy = np.zeros((opt_count))
                        for opt in range(opt_count):
                            calibrated_option_entropy[opt] = cal_opt_freq[opt] * np.log(
                                cal_opt_freq[opt]) if cal_opt_freq[opt] != 0 else 0
                            baseline_option_entropy[opt] = base_opt_freq[opt] * np.log(
                                base_opt_freq[opt]) if base_opt_freq[opt] != 0 else 0
                        calibrated_sample_entropy = 1 - ((-1 / np.log(opt_count)) *
                                                         np.sum(calibrated_option_entropy))
                        baseline_sample_entropy = 1 - ((-1 / np.log(opt_count)) * np.sum(baseline_option_entropy))

                        calibrated_entropies.append(calibrated_sample_entropy)
                        baseline_entropies.append(baseline_sample_entropy)

                    # Write to CSV
                    writerow = {
                        "index": i,
                        "image_path": image_path,
                        "question": question,
                        "gt_answer": gt_answer,
                        "calibrated_answer": calibrated_answer,
                        "calibrated_prob": calibrated_prob,
                        "calibrated_output": calibrated_output,
                        "baseline_answer": baseline_answer,
                        "baseline_prob": baseline_prob,
                        "baseline_output": baseline_output
                    }
                    if self.args.entropy:
                        writerow["calibrated_entropy"] = calibrated_sample_entropy
                        writerow["baseline_entropy"] = baseline_sample_entropy

                    csv_writer.writerow(writerow)

                    # Ensure data is written to disk
                    csv_file.flush()
                    os.fsync(csv_file.fileno())

                except Exception as e:
                    print(f"Error on index {i}: {e}")
                    continue

        csv_file.close()
        logs_df = pd.read_csv(csv_path)

        # 4. Calcute metrics
        gt_answers = logs_df['gt_answer']
        calibrated_lm_answers = logs_df['calibrated_answer']
        calibrated_lm_probabilities = logs_df['calibrated_prob']
        baseline_lm_answers = logs_df['baseline_answer']
        baseline_lm_probabilities = logs_df['baseline_prob']

        calibrated_acc = compute_accuracy(
            gt_answers, calibrated_lm_answers)
        calibrated_ece = compute_ece(
            gt_answers, calibrated_lm_answers, calibrated_lm_probabilities)
        calibrated_brier = compute_brier_score(
            gt_answers, calibrated_lm_answers, calibrated_lm_probabilities)
        calibrated_confidence_avg = np.mean(calibrated_lm_probabilities)
        calibrated_confidence_std = np.std(calibrated_lm_probabilities)

        baseline_acc = compute_accuracy(
            gt_answers, baseline_lm_answers)
        baseline_ece = compute_ece(
            gt_answers, baseline_lm_answers, baseline_lm_probabilities)
        baseline_brier = compute_brier_score(
            gt_answers, baseline_lm_answers, baseline_lm_probabilities)
        baseline_confidence_avg = np.mean(baseline_lm_probabilities)
        baseline_confidence_std = np.std(baseline_lm_probabilities)

        # filter for valid samples only (i.e answer != -1 and conf != 1e-12)
        valid_mask_calibrated = filter_valid_samples(
            gt_answers, calibrated_lm_answers, calibrated_lm_probabilities)
        valid_calibrated_count = int(np.sum(valid_mask_calibrated))

        valid_mask_baseline = filter_valid_samples(
            gt_answers, baseline_lm_answers, baseline_lm_probabilities)
        valid_baseline_count = int(np.sum(valid_mask_baseline))

        # valid samples only metrics
        calibrated_acc_valid = compute_accuracy(
            np.array(gt_answers)[valid_mask_calibrated],
            np.array(calibrated_lm_answers)[valid_mask_calibrated]
        )
        calibrated_ece_valid = compute_ece(
            np.array(gt_answers)[valid_mask_calibrated],
            np.array(calibrated_lm_answers)[valid_mask_calibrated],
            np.array(calibrated_lm_probabilities)[valid_mask_calibrated]
        )
        calibrated_brier_valid = compute_brier_score(
            np.array(gt_answers)[valid_mask_calibrated],
            np.array(calibrated_lm_answers)[valid_mask_calibrated],
            np.array(calibrated_lm_probabilities)[valid_mask_calibrated]
        )
        calibrated_confidence_avg_valid = np.mean(
            np.array(calibrated_lm_probabilities)[valid_mask_calibrated]
        )
        calibrated_confidence_std_valid = np.std(
            np.array(calibrated_lm_probabilities)[valid_mask_calibrated]
        )

        baseline_acc_valid = compute_accuracy(
            np.array(gt_answers)[valid_mask_baseline],
            np.array(baseline_lm_answers)[valid_mask_baseline]
        )
        baseline_ece_valid = compute_ece(
            np.array(gt_answers)[valid_mask_baseline],
            np.array(baseline_lm_answers)[valid_mask_baseline],
            np.array(baseline_lm_probabilities)[valid_mask_baseline]
        )
        baseline_brier_valid = compute_brier_score(
            np.array(gt_answers)[valid_mask_baseline],
            np.array(baseline_lm_answers)[valid_mask_baseline],
            np.array(baseline_lm_probabilities)[valid_mask_baseline]
        )
        baseline_confidence_avg_valid = np.mean(
            np.array(baseline_lm_probabilities)[valid_mask_baseline]
        )
        baseline_confidence_std_valid = np.std(
            np.array(baseline_lm_probabilities)[valid_mask_baseline]
        )

        results = {
            "calibrated_accuracy": calibrated_acc,
            "calibrated_ece": calibrated_ece,
            "calibrated_brier": calibrated_brier,
            "calibrated_confidence_avg": calibrated_confidence_avg,
            "calibrated_confidence_std": calibrated_confidence_std,

            "baseline_accuracy": baseline_acc,
            "baseline_ece": baseline_ece,
            "baseline_brier": baseline_brier,
            "baseline_confidence_avg": baseline_confidence_avg,
            "baseline_confidence_std": baseline_confidence_std,

            "calibrated_valid_count": valid_calibrated_count,
            "calibrated_accuracy_valid": calibrated_acc_valid,
            "calibrated_ece_valid": calibrated_ece_valid,
            "calibrated_brier_valid": calibrated_brier_valid,
            "calibrated_confidence_avg_valid": calibrated_confidence_avg_valid,
            "calibrated_confidence_std_valid": calibrated_confidence_std_valid,

            "baseline_valid_count": valid_baseline_count,
            "baseline_accuracy_valid": baseline_acc_valid,
            "baseline_ece_valid": baseline_ece_valid,
            "baseline_brier_valid": baseline_brier_valid,
            "baseline_confidence_avg_valid": baseline_confidence_avg_valid,
            "baseline_confidence_std_valid": baseline_confidence_std_valid,

            "checkpoint": self.checkpoint_path,
        }

        # 5. Log Metrics
        if self.args.entropy:
            calibrated_entropy_mean = np.nanmean(logs_df["calibrated_entropy"])
            baseline_entropy_mean = np.nanmean(logs_df["baseline_entropy"])

            results["calibrated_entropy_mean"] = calibrated_entropy_mean
            results["baseline_entropy_mean"] = baseline_entropy_mean

        with open(log_file, "w") as f:
            json.dump(results, f, indent=4)

        print(f"Metric Results saved to {log_file}")

    def eval_stable(self):
        raise NotImplementedError('Deprecated eval_stable(). Use eval() instead.')
        # 1. Load model and tokenizer from checkpoint
        print(f"Loading model from checkpoint: {self.checkpoint_path}")
        model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint_path).to("cuda")
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
        calibrated_entropies = []
        baseline_lm_answers = []
        baseline_lm_probabilities = []
        baseline_entropies = []
        for sample in tqdm(eval_dataset):
            question = sample["question"]
            options = sample["options"]
            gt_answer = sample["gt_answer"]
            prompt = sample["prompt"]

            if self.config.dataset.name == 'pmcvqa':
                image_path = sample["image_path"]
                absolute_image_path = os.path.abspath(image_path)
                assert absolute_image_path.startswith(
                    self.config.dataset.image_root)
                assert os.path.exists(absolute_image_path)
                image_url = f"file://{absolute_image_path}"

            inputs = tokenizer(prompt[0]["content"],
                               return_tensors="pt").to("cuda")

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=self.config.train.max_completion_length,
                    do_sample=False  # Use greedy decoding during evaluation for deterministic results
                )
            completion = tokenizer.decode(
                output[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
            calibrated_prompt = build_downstream_prompt(
                dataset_name=self.config.dataset.name,
                question_text=question,
                option_list=options,
                hint_text=completion,
            )

            if self.config.dataset.name == 'medmcqa':
                calibrated_output = self.send_message_downstream(
                    {'role': 'user', 'content': calibrated_prompt}, seed=1)
                calibrated_answer, calibrated_prob = parse_answer_prob(
                    calibrated_output)

            elif self.config.dataset.name == 'pmcvqa':
                calibrated_output = self.send_message_downstream(
                    {
                        'role': 'user',
                        'content': [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url,
                                }
                            },
                            {
                                "type": "text",
                                "text": calibrated_prompt,
                            }
                        ]
                    },
                    seed=1
                )

                calibrated_answer, calibrated_prob = parse_answer_prob_vqa(
                    calibrated_output)

            else:
                raise ValueError(
                    'Invalid dataset. Only medmcqa and pmcvqa are supported.')

            calibrated_lm_answers.append(calibrated_answer)
            calibrated_lm_probabilities.append(calibrated_prob)

            baseline_prompt = build_downstream_prompt(
                dataset_name=self.config.dataset.name,
                question_text=question,
                option_list=options,
                hint_text=None,
            )

            if self.config.dataset.name == 'medmcqa':
                baseline_output = self.send_message_downstream(
                    {'role': 'user', 'content': baseline_prompt}, seed=1)
                baseline_answer, baseline_prob = parse_answer_prob(
                    baseline_output)

            elif self.config.dataset.name == 'pmcvqa':
                baseline_output = self.send_message_downstream(
                    {
                        'role': 'user',
                        'content': [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url,
                                }
                            },
                            {
                                "type": "text",
                                "text": baseline_prompt,
                            }
                        ]
                    },
                    seed=1
                )

                baseline_answer, baseline_prob = parse_answer_prob_vqa(
                    baseline_output)

            else:
                raise ValueError(
                    'Invalid dataset. Only medmcqa and pmcvqa are supported.')

            baseline_lm_answers.append(baseline_answer)
            baseline_lm_probabilities.append(baseline_prob)

            if self.args.entropy:
                # Implement uncertainty quantification, referencing: [1] Q.
                # Lyu et al., “Calibrating Large Language Models with Sample Consistency”.
                # Use default of 40 MC samples (consistent with Lyu)
                mc_samples = 40

                opt_count = len(options)

                cal_opt_freq = np.zeros((opt_count))
                base_opt_freq = np.zeros((opt_count))
                for mc_sample in tqdm(range(mc_samples)):

                    if self.config.dataset.name == 'medmcqa':
                        calibrated_output = self.send_message_downstream(
                            {'role': 'user', 'content': calibrated_prompt})
                        baseline_output = self.send_message_downstream(
                            {'role': 'user', 'content': baseline_prompt})
                        calibrated_answer, calibrated_prob = parse_answer_prob(
                            calibrated_output)
                        baseline_answer, baseline_prob = parse_answer_prob(
                            baseline_output)

                    elif self.config.dataset.name == 'pmcvqa':
                        calibrated_output = self.send_message_downstream(
                            {
                                'role': 'user',
                                'content': [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": image_url,
                                        }
                                    },
                                    {
                                        "type": "text",
                                        "text": calibrated_prompt,
                                    }
                                ]
                            }
                        )

                        baseline_output = self.send_message_downstream(
                            {
                                'role': 'user',
                                'content': [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": image_url,
                                        }
                                    },
                                    {
                                        "type": "text",
                                        "text": baseline_prompt,
                                    }
                                ]
                            }
                        )

                        baseline_answer, baseline_prob = parse_answer_prob_vqa(
                            baseline_output)
                        calibrated_answer, calibrated_prob = parse_answer_prob_vqa(
                            calibrated_output)
                        baseline_answer, baseline_prob = parse_answer_prob_vqa(
                            baseline_output)

                        calibrated_answer = convert_letter_to_idx(
                            calibrated_answer)
                        baseline_answer = convert_letter_to_idx(
                            baseline_answer)

                    # Don't count invalid responses (-1 or non-int format)
                    if type(calibrated_answer) == int and calibrated_answer >= 0 and calibrated_answer - 1 < opt_count:
                        # Subtract 1 for indexing since multiple choice answer numbers are not zero-indexed
                        cal_opt_freq[calibrated_answer - 1] += 1

                    # Don't count invalid responses (-1 or non-int format)
                    if type(baseline_answer) == int and baseline_answer >= 0 and baseline_answer - 1 < opt_count:
                        # Subtract 1 for indexing since multiple choice answer numbers are not zero-indexed
                        base_opt_freq[baseline_answer - 1] += 1

                # Normalize frequencies based on number of valid answers
                cal_opt_freq = cal_opt_freq / np.sum(cal_opt_freq)
                base_opt_freq = base_opt_freq / np.sum(base_opt_freq)

                calibrated_option_entropy = np.zeros((opt_count))
                baseline_option_entropy = np.zeros((opt_count))
                for opt in range(opt_count):
                    calibrated_option_entropy[opt] = cal_opt_freq[opt] * np.log(
                        cal_opt_freq[opt]) if cal_opt_freq[opt] != 0 else 0
                    baseline_option_entropy[opt] = base_opt_freq[opt] * np.log(
                        base_opt_freq[opt]) if base_opt_freq[opt] != 0 else 0
                calibrated_sample_entropy = 1 - ((-1 / np.log(opt_count)) * np.sum(calibrated_option_entropy))
                baseline_sample_entropy = 1 - ((-1 / np.log(opt_count)) * np.sum(baseline_option_entropy))

                print(f'CALIBRATED ENTROPY: {calibrated_sample_entropy, cal_opt_freq}')
                print(f'BASELINE ENTROPY: {baseline_sample_entropy, base_opt_freq}')

                calibrated_entropies.append(calibrated_sample_entropy)
                baseline_entropies.append(baseline_sample_entropy)

        # 4. Calcute metrics
        calibrated_acc = compute_accuracy(
            eval_dataset["gt_answer"], calibrated_lm_answers)
        calibrated_ece = compute_ece(
            eval_dataset["gt_answer"], calibrated_lm_answers, calibrated_lm_probabilities)
        calibrated_brier = compute_brier_score(
            eval_dataset["gt_answer"], calibrated_lm_answers, calibrated_lm_probabilities)
        calibrated_confidence_avg = np.mean(calibrated_lm_probabilities)
        calibrated_confidence_std = np.std(calibrated_lm_probabilities)

        baseline_acc = compute_accuracy(
            eval_dataset["gt_answer"], baseline_lm_answers)
        baseline_ece = compute_ece(
            eval_dataset["gt_answer"], baseline_lm_answers, baseline_lm_probabilities)
        baseline_brier = compute_brier_score(
            eval_dataset["gt_answer"], baseline_lm_answers, baseline_lm_probabilities)
        baseline_confidence_avg = np.mean(baseline_lm_probabilities)
        baseline_confidence_std = np.std(baseline_lm_probabilities)

        # 5. Log Metrics
        if self.args.entropy:
            calibrated_entropy_mean = np.nanmean(calibrated_entropies)
            baseline_entropy_mean = np.nanmean(baseline_entropies)

            results = {
                "calibrated_accuracy": calibrated_acc,
                "calibrated_ece": calibrated_ece,
                "calibrated_brier": calibrated_brier,
                "calibrated_confidence_avg": calibrated_confidence_avg,
                "calibrated_confidence_std": calibrated_confidence_std,
                "calibrated_entropy_mean": calibrated_entropy_mean,
                "baseline_accuracy": baseline_acc,
                "baseline_ece": baseline_ece,
                "baseline_brier": baseline_brier,
                "baseline_confidence_avg": baseline_confidence_avg,
                "baseline_confidence_std": baseline_confidence_std,
                "baseline_entropy_mean": baseline_entropy_mean,
                "checkpoint": self.checkpoint_path,
            }
        else:
            results = {
                "calibrated_accuracy": calibrated_acc,
                "calibrated_ece": calibrated_ece,
                "calibrated_brier": calibrated_brier,
                "calibrated_confidence_avg": calibrated_confidence_avg,
                "calibrated_confidence_std": calibrated_confidence_std,
                "baseline_accuracy": baseline_acc,
                "baseline_ece": baseline_ece,
                "baseline_brier": baseline_brier,
                "baseline_confidence_avg": baseline_confidence_avg,
                "baseline_confidence_std": baseline_confidence_std,
                "checkpoint": self.checkpoint_path,
            }

        log_file = os.path.join(self.log_path, "eval_results.json")
        os.makedirs(self.log_path, exist_ok=True)

        with open(log_file, "w") as f:
            json.dump(results, f, indent=4)

        print(f"Metric Results saved to {log_file}")
