import sys
import argparse
import os
import yaml
import torch
import numpy as np
from engine.agent import Agent
from engine import wait_until_ready
import util
import subprocess


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
    parser.add_argument("--component", type=int, default=-1, choices=[-1, 0, 1],
                        help="Running on which component:\n0: vLLM for TRL action sampling AND vLLM for downstream\n1: TRL for policy update/training")

    args = parser.parse_args()
    # check if only one is True from "train", "ctrain", "test"
    if sum([args.train, args.ctrain, args.test]) != 1:
        raise ValueError("Exactly one of --train, --ctrain, or --test must be specified.")

    # parse config file
    with open(os.path.join("config", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    # add argument for closed source downstream
    args.is_closed_source_downstream = util.is_closed_source_model(new_config.model.downstream)

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
    args, config = parse_args_and_config()
    if args.train:
        if args.component == 0:
            env1 = os.environ.copy()
            # deploy vLLM for TRL action sampling, use by "python main.py"
            env1["VLLM_USE_V1"] = "0"
            env1["CUDA_VISIBLE_DEVICES"] = config.resources.action_cuda
            num_gpus = len(config.resources.action_cuda.split(","))
            env1["XDG_CACHE_HOME"] = config.resources.cache_dir
            # run trl vllm serve
            subprocess.Popen(["trl",
                              "vllm-serve",
                              f"--model={config.model.policy}",
                              f"--gpu_memory_utilization={config.resources.action_gpu_memory_utilization}",
                              f"--tensor_parallel_size={num_gpus}",
                              f"--port={config.resources.action_port}"],
                             env=env1,
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL,
                             start_new_session=True)

            # deploy vLLM for downstream, use by "python main.py"
            if not args.is_closed_source_downstream:
                env2 = os.environ.copy()
                env2["CUDA_VISIBLE_DEVICES"] = config.resources.downstream_cuda
                num_gpus = len(config.resources.downstream_cuda.split(","))
                env2["XDG_CACHE_HOME"] = config.resources.cache_dir
                # run vllm serve
                subprocess.Popen(["vllm",
                                  "serve",
                                  config.model.downstream,
                                  f"--gpu_memory_utilization={config.resources.downstream_gpu_memory_utilization}",
                                  f"--tensor_parallel_size={num_gpus}",
                                  f"--port={config.resources.downstream_port}"],
                                 env=env2,
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL,
                                 start_new_session=True)
        elif args.component == 1:  # deploy TRL for policy update/training, use by "accelerate launch main.py"
            if not args.is_closed_source_downstream:
                # check if two ports are available for querying
                util.info('main.py', 'Waiting for vLLM to be ready (for downstream)...')
                wait_until_ready(port=config.resources.downstream_port)
            util.info('main.py', 'Waiting for TRL vLLM-Serve to be ready (for action sampling)...')
            wait_until_ready(port=config.resources.action_port)
            util.info('main.py', 'vLLMs are ready!')
            os.environ["CUDA_VISIBLE_DEVICES"] = config.resources.policy_cuda
            os.environ["XDG_CACHE_HOME"] = config.resources.cache_dir
            agent = Agent(args, config)
            agent.train()
        else:
            raise ValueError("Invalid component specified. Choose from 0, 1, or 2.")
    elif args.ctrain:
        pass
    elif args.test:
        pass


if __name__ == "__main__":
    sys.exit(main())
