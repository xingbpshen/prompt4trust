from trl import GRPOConfig, GRPOTrainer


class Agent:
    def __init__(self, policy_model):
        self.policy_model = policy_model
