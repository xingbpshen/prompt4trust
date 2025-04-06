import sys
import os
from engine import wait_until_ready
import util
import subprocess


def main():
    args_to_forward = sys.argv[1:]
    args, config = util.parse_args_and_config()
    if args.train:
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

        # wait for vLLM to be ready
        if not args.is_closed_source_downstream:
            # check if two ports are available for querying
            util.info('main.py', 'Waiting for vLLM to be ready (for downstream)...')
            wait_until_ready(port=config.resources.downstream_port)
        util.info('main.py', 'Waiting for TRL vLLM-Serve to be ready (for action sampling)...')
        wait_until_ready(port=config.resources.action_port)
        util.info('main.py', 'vLLMs are ready!')

        # deploy TRL for policy update/training, use by "accelerate launch main.py"
        env3 = os.environ.copy()
        env3["CUDA_VISIBLE_DEVICES"] = config.resources.policy_cuda
        env3["XDG_CACHE_HOME"] = config.resources.cache_dir
        subprocess.Popen(["accelerate", "launch", "engine_launcher.py"] + args_to_forward,
                         env=env3, start_new_session=False)
    elif args.ctrain:
        pass
    elif args.test:
        pass


if __name__ == "__main__":
    sys.exit(main())
