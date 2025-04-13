import json
import matplotlib.pyplot as plt
import numpy as np
import os

def moving_average(x, window_size=20):
    return np.convolve(x, np.ones(window_size)/window_size, mode='same')

def clip_values(x, min_val, max_val):
    return np.clip(x, min_val, max_val)

input_path = '/network/scratch/a/anita.kriz/vccrl-llm/test/test_1/checkpoint-3000/trainer_state.json' #TODO
output_path = '/network/scratch/a/anita.kriz/vccrl-llm/plots/metrics.png' #TODO

with open(input_path, 'r') as f:
    data = json.load(f)

log_history = data["log_history"]

steps = [entry["step"] for entry in log_history]
losses = [entry["loss"] for entry in log_history]
rewards = [entry["reward"] for entry in log_history]
reward_stds = [entry["reward_std"] for entry in log_history]
kls = [entry["kl"] for entry in log_history]
grad_norm = [entry["grad_norm"] for entry in log_history]
clip_ratio = [entry["clip_ratio"] for entry in log_history]

# Smooth & clip loss and kl
losses_smooth = clip_values(moving_average(losses), -10, 1)
kls_smooth = clip_values(moving_average(kls), 0, 5)
grad_norm = clip_values(moving_average(grad_norm), 0, 1000)

fig, axs = plt.subplots(3, 2, figsize=(12, 8))

# Reward
axs[0, 0].plot(steps, rewards, label='Reward', color='green')
axs[0, 0].fill_between(steps, 
                      np.array(rewards)-np.array(reward_stds), 
                      np.array(rewards)+np.array(reward_stds), 
                      color='green', alpha=0.2)
axs[0, 0].set_title('train/reward')

# Reward Std
axs[0, 1].plot(steps, reward_stds, label='Reward Std', color='blue')
axs[0, 1].set_title('train/reward_std')

# KL
axs[1, 0].plot(steps, kls_smooth, label='KL', color='orange')
axs[1, 0].set_title('train/kl')

# Loss
axs[1, 1].plot(steps, losses_smooth, label='Loss', color='red')
axs[1, 1].set_title('train/loss')

# Grad Norm
axs[2, 0].plot(steps, grad_norm, label='Grad Norm', color='purple')
axs[2, 0].set_title('train/grad_norm')

# Clip Ratio
axs[2, 1].plot(steps, clip_ratio, label='Clip Ratio', color='violet')
axs[2, 1].set_title('train/clip_ratio')

for ax in axs.flat:
    ax.set_xlabel('Step')
    ax.grid(True)

plt.tight_layout()
plt.savefig(output_path)
print(f'Plot saved to {output_path}')

