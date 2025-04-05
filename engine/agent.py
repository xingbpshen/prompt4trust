from trl import GRPOConfig, GRPOTrainer
from engine import init_log_path, reward_func
import os
import dataset


class Agent:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.log_path = os.path.join(args.log_folder, args.trial_name)
        init_log_path(self.log_path, args)

    def train(self, trainer_name='GRPO'):
        assert trainer_name == 'GRPO'
        # minimum example
        trainer_config = GRPOConfig(output_dir=str(self.log_path),
                                    logging_steps=self.config.train.logging_steps,
                                    temperature=self.config.train.temperature,
                                    top_p=self.config.train.top_p,
                                    top_k=self.config.train.top_k,
                                    use_vllm=self.config.train.use_vllm,
                                    learning_rate=self.config.train.learning_rate,
                                    scale_rewards=self.config.train.scale_rewards)
        trainer = GRPOTrainer(model=self.config.model.policy,
                              reward_funcs=reward_func,
                              args=trainer_config,
                              train_dataset=dataset.get_dataset(args=self.args, config=self.config, split='train'))
