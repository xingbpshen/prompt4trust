<h1 align="center">
Prompt4Trust
</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2507.09279"><img src="https://img.shields.io/badge/arXiv-2507.09279-b31b1b.svg" alt="arXiv"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python Version"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.6.0-red.svg" alt="PyTorch Version"></a>
</p>

This repository contains the official implementation of the paper:
> __Prompt4Trust: A Reinforcement Learning Prompt Augmentation Framework for Clinically-Aligned Confidence Calibration in Multimodal Large Language Models__  
> [Anita Kriz*](https://scholar.google.com/citations?view_op=list_works&hl=en&hl=en&user=EDKX_QgAAAAJ), [Elizabeth Laura Janes*](https://scholar.google.com/citations?user=O41pk_EAAAAJ&hl=en), [Xing Shen*](https://scholar.google.com/citations?hl=en&user=U69NqfQAAAAJ), [Tal Arbel](https://www.cim.mcgill.ca/~arbel/)  
> _*Equal contribution_  
> _IEEE/CVF International Conference on Computer Vision 2025 Workshop CVAMD_  
> __Paper ([arXiv preprint](https://arxiv.org/abs/2507.09279))__


## Before you start
Make sure you have at least **4** NVIDIA GPUs with adequate memory (memory requirement depends on the scale of the LLM/MLLM you want to use) if you wish to use open-source downstream task MLLMs.

### Downloading open-source LLMs/MLLMs beforehand
We recommend to download the open-source LLMs/MLLMs using `huggingface-cli` before you start (make sure you obtained relevant permissions/agreement to download the models from Hugging Face):
```bash
huggingface-cli login
huggingface-cli download {REPO_NAME} --local-dir {SAVE_FOLDER} --local-dir-use-symlinks False
```
For example, the `{REPO_NAME}` can be `Qwen/Qwen2.5-1.5B-Instruct` and `{SAVE_FOLDER}` can be `/usr/local/data/Qwen2.5-1.5B-Instruct`. The downloaded model will be saved in the specified folder `{SAVE_FOLDER}`.

## 1. Preparation

### 1.1 Installation
It is recommended to use a virtual environment (e.g., `venv`) to avoid package conflicts. Here we assume you are using `venv` as your virtual environment. If you are using `conda`, please adjust the commands accordingly.
```bash
git clone https://github.com/xingbpshen/prompt4trust.git
cd prompt4trust/
pip install -r requirements.txt
```

### 1.2 Downloading datasets
Create a `data/` folder under the project root `prompt4trust/`:
```bash
mkdir data/
```
Download the dataset from the PMC-VQA repository [here](https://huggingface.co/datasets/RadGenome/PMC-VQA). Put all files in the newly created `data/` folder.
Our training split can be generated using `dataset/gen_train.py` (with modification to data path) or can be downloaded here (coming soon).

### 1.3 Modifying config
The config files are located at `config/`. You can modify the parameters according to your needs. The default config file `pmcvqa.yml` is for the PMC-VQA dataset.

Here are some important parameters you may want to modify:
- `resources.cache_dir` This is where vLLM and other python packages will be cached. Make sure you have enough space.
- `resources.policy_cuda` This is a string of CUDA devices (e.g., `"3,4"` or `"3"`) used for the policy update/training. Make sure you have enough memory on these devices.
- `resources.action_cuda` This is a string of CUDA devices used for the TRL with vLLM serving to sample "actions" (in the context of reinforcement learning). Make sure you have enough memory on these devices.
- `resources.downstream_cuda` This is a string of CUDA devices used for the downstream MLLM (to obtain reward). Make sure you have enough memory on these devices.
- `model.policy` This is the model name. You can use any repository name supported by Hugging Face or a path to a local model (e.g., `"Qwen/Qwen2.5-1.5B-Instruct"` or `"/usr/local/data/Qwen2.5-1.5B-Instruct"`).
- `model.downstream` This is the model name. You can use any repository name supported by Hugging Face or a path to a local model.

Please note that `resources.policy_cuda`, `resources.action_cuda`, and `resources.downstream_cuda` **must not include any overlapping device** to avoid CUDA initialization error.

## 2. Training
### 2.1 About vLLM serving
To enable TRL with vLLM serving, we need to start **2** servers: one for the policy model (to sample action) and one for the downstream LLM to calculate reward.

These servers will be started automatically so you do not need to do anything now.
### 2.2 Training the policy model
By default, the policy model will be trained with GRPO using TRL support. Run the following command to start training:
```bash
python main.py --config {DATASET}.yml --log_folder {LOG_FOLDER} --trial_name {TRIAL_NAME} --train --ni
```
Running the above command once will start:
- **2** detached subprocesses for vLLMs, each corresponding to one of the servers. You can observe the GPU memory usage increasing in the terminal. You can use `nvidia-smi` to check the GPU memory usage for your specified CUDA devices `resources.action_cuda` and `resources.downstream_cuda`.
- **1** foreground engine subprocess for TRL, which will be responsible for the training of the policy model. You can observe the GPU memory usage (on your specified CUDA devices `resources.policy_cuda`) increasing in the terminal.

Runtime related logs will be saved in `{LOG_FOLDER}/{TRIAL_NAME}/` folder.

## 3. Evaluation
Run the following command to evaluate the trained policy model:
```bash
python main.py --config {DATASET}.yml --log_folder {LOG_FOLDER} --trial_name {TRIAL_NAME} --test --ni 
```

## Acknowledgements
This work was supported in part by the Natural Sciences and Engineering Research Council of Canada, in part by the Canadian Institute for Advanced Research (CIFAR) Artificial Intelligence Chairs Program, in part by the Mila—Quebec Artificial Intelligence Institute, in part by the Mila—Google Research Grant, in part by the Fonds de recherche du Québec, in part by the Canada First Research Excellence Fund, awarded to the Healthy Brains, Healthy Lives initiative at McGill University, and in part by the Department of Electrical and Computer Engineering at McGill University.

## Contact
Please raise a GitHub issue or email us at <a href="mailto:xing.shen@mail.mcgill.com">xing.shen@mail.mcgill.com</a> (with the email subject starting with "[Prompt4Trust]") if you have any question or encounter any issue.
