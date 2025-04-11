from huggingface_hub import repo_exists
import os
import argparse
import yaml
import numpy as np
import torch
import psutil
import warnings

def info(file_name, msg):
    print(f"\033[1;94m[{file_name}]\033[0m \033[94mINFO\033[0m {msg}")


def is_closed_source_model(model_name):
    # it is a closed source model if os does not exist model_name path or it is not available on huggingface
    if os.path.exists(model_name):
        return False
    elif repo_exists(model_name):
        return False
    else:
        return True

# check GPU compute capability
def get_gpu_compute_capability():
    if torch.cuda.is_available():
        return torch.cuda.get_device_capability(0)
    else:
        warnings.warn("No CUDA-compatible GPU found.", UserWarning)
        return None

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, required=True, help="Name of the config file (e.g., medmcqa.yml)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--log_folder", type=str, default="./log", help="Path for saving running related data, e.g., ./log"
    )
    parser.add_argument(
        "--trial_name",
        type=str,
        required=True,
        help="A string for documentation purpose. "
             "Will be the name of the folder inside the log folder and the comet trial name.",
    )
    parser.add_argument("--train", action="store_true", help="Whether to train")
    parser.add_argument("--ctrain", action="store_true", help="Whether to continue training")
    parser.add_argument("--test", action="store_true", help="Whether to test")
    parser.add_argument(
        "--comment", type=str, default="", help="A string for experiment comment"
    )
    parser.add_argument("--ni", action="store_true",
                        help="Whether to use no interaction mode (automatically accept all prompts)")
    parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint') #add arguement for resume

    args = parser.parse_args()
    # check arg legitimacy
    if sum([args.train, args.ctrain, args.test]) != 1:
        raise ValueError("Exactly one of --train, --ctrain, or --test must be specified.")
    assert args.ni is True

    # parse config file
    with open(os.path.join("config", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    # add argument for closed source downstream
    args.is_closed_source_downstream = is_closed_source_model(new_config.model.downstream)

    #add argument for compute capability 
    args.compute_capability = get_gpu_compute_capability()

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace
