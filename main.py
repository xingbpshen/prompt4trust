import sys
import argparse
import os
import yaml
import torch
import numpy as np


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--log_folder", type=str, default="./log", help="Path for saving running related data, e.g., ./log."
    )
    parser.add_argument(
        "--trial_name",
        type=str,
        required=True,
        help="A string for documentation purpose. "
        "Will be the name of the folder inside the log folder and the comet trial name.",
    )
    parser.add_argument(
        "--comment", type=str, default="", help="A string for experiment comment"
    )
    default_devices = ",".join(str(i) for i in range(torch.cuda.device_count()))
    parser.add_argument(
        "--cuda_devices",
        type=str,
        default=default_devices,
        help="Comma-separated list of CUDA devices visible (e.g., 0,1,2,3,4)"
    )

    args = parser.parse_args()

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

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


def main():
    pass


if __name__ == "__main__":
    sys.exit(main())
