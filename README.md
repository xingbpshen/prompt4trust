<h1 align="center">
Verbalized Confidence Calibration in Large Language Models with Reinforcement Learning
</h1>

<p align="center">
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python Version">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-2.6.0-red.svg" alt="PyTorch Version">
  </a>
</p>

# Work in Progress

## 0. Before you start
Make sure you have at least **3** NVIDIA GPUs with adequate memory (memory requirement depends on the scale of the LLM you want to use).

## 1. Preparation
### 1.1 Installation
```bash
pip install -r requirements.txt
```
### 1.2 Downloading datasets
Create a `data/` folder under the project root:
```bash
cd vccrl-llm/
mkdir data/
```
Download the dataset from the MedMCQA repository [here](https://drive.google.com/uc?export=download&id=15VkJdq5eyWIkfb_aoD3oS8i4tScbHYky). Unzip and put all three `.json` files in the newly created `data/` folder.
### 1.3 Modifying config
The config files are located at `config/`. You can modify the parameters according to your needs. The default config file `medmcqa.yml` is for the MedMCQA dataset.

Here are some important parameters you may want to modify:
- `resources.cache_dir` This is where vLLM and other python packages will be cached. Make sure you have enough space.
- `resources.policy_cuda` This is a string of CUDA devices (e.g., `"3,4,5"` or `"3"`) used for the policy update/training. Make sure you have enough memory on these devices.
- `resources.action_cuda` This is a string of CUDA devices used for the TRL with vLLM serving to sample "actions" (in the context of reinforcement learning). Make sure you have enough memory on these devices.
- `resources.downstream_cuda` This is a string of CUDA devices used for the downstream LLM (to obtain reward). Make sure you have enough memory on these devices.
- `model.policy` and `model.downstream` are the model names. You can use any model name supported by Hugging Face or a path to a local model (e.g., `"meta-llama/Llama-3.1-8B-Instruct"` or `"/usr/local/data/Llama-3.1-8B-Instruct"`).

## 2. Training
### 2.1 Starting two servers
To enable TRL with vLLM serving, you need to start two servers: one for the policy model (to sample action) and one for the downstream LLM to calculate reward. You can use the following commands to start the servers:
```bash
python main.py --config {DATASET}.yml --log_folder {LOG_FOLDER} --trial_name {TRIAL_NAME} --train --component 0
```
Runtime related logs will be saved in `{LOG_FOLDER}/{TRIAL_NAME}/` folder.

Running the above command once will start two detached subprocesses, each corresponding to one of the servers. You can observe the GPU memory usage increasing in the terminal. You can use `nvidia-smi` to check the GPU memory usage for your specified CUDA devices `resources.action_cuda` and `resources.downstream_cuda`.
### 2.2 Training the policy model
By default, the policy model will be trained with GRPO using TRL support. Run the following command to start training:
```bash
accelerate launch main.py --config {DATASET}.yml --log_folder {LOG_FOLDER} --trial_name {TRIAL_NAME} --train --component 1
```
Please test it out and let me know (by raising an GitHub issue) if you encounter any issues.