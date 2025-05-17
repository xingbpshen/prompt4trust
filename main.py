import sys
import os
from engine import wait_until_ready
import util
import subprocess
import signal


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

        if args.compute_capability is not None:
            if args.compute_capability[0] >= 8:  # for GPUs with compute capability >= 8.0, can use bfloat16
                dtype = "bfloat16"
            else:  # for older GPUs, must use float16
                dtype = "half"
        else:
            dtype = "half"  # default to half if no GPU is found
    
        # run trl vllm serve
        action_proc = subprocess.Popen(["trl",
                                        "vllm-serve",
                                        f"--model={config.model.policy}",
                                        f"--gpu_memory_utilization={config.resources.action_gpu_memory_utilization}",
                                        f"--tensor_parallel_size={num_gpus}",
                                        f"--host=localhost",
                                        f"--port={config.resources.action_port}",
                                        f"--dtype={dtype}"],
                                       env=env1,
                                       stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL,
                                       start_new_session=True)

        # deploy vLLM for downstream, use by "python main.py"
        downstream_proc = None
        if not args.is_closed_source_downstream:
            env2 = os.environ.copy()
            env2["CUDA_VISIBLE_DEVICES"] = config.resources.downstream_cuda
            num_gpus = len(config.resources.downstream_cuda.split(","))
            env2["XDG_CACHE_HOME"] = config.resources.cache_dir
            # run vllm serve
            command = [
                "vllm", "serve", config.model.downstream,
                f"--gpu_memory_utilization={config.resources.downstream_gpu_memory_utilization}",
                f"--tensor_parallel_size={num_gpus}",
                "--host=localhost",
                f"--port={config.resources.downstream_port}",
                f"--dtype={dtype}",
                f"--allowed-local-media-path={config.dataset.image_root}"
            ]
            util.info('main.py', f"Launching vLLM with command: {' '.join(command)}")

            downstream_proc = subprocess.Popen(["vllm",
                                                "serve",
                                                config.model.downstream,
                                                f"--gpu_memory_utilization={config.resources.downstream_gpu_memory_utilization}",
                                                f"--tensor_parallel_size={num_gpus}",
                                                f"--host=localhost",
                                                f"--port={config.resources.downstream_port}",
                                                f"--allowed-local-media-path={config.dataset.image_root}",
                                                f"--dtype={dtype}"],
                                               env=env2,
                                               stdout=subprocess.DEVNULL,
                                               stderr=subprocess.DEVNULL,
                                               start_new_session=True)

        # wait for vLLM to be ready
        if not args.is_closed_source_downstream:
            # check if two ports are available for querying
            util.info('main.py', 'Waiting for vLLM to be ready (for downstream)...')
            wait_until_ready(port=config.resources.downstream_port, subproc=downstream_proc)
        util.info('main.py', 'Waiting for TRL vLLM-Serve to be ready (for action sampling)...')
        wait_until_ready(port=config.resources.action_port, subproc=action_proc)
        util.info('main.py', 'vLLMs are ready!')

        # deploy TRL for policy update/training, use by "accelerate launch main.py"
        env3 = os.environ.copy()
        env3["CUDA_VISIBLE_DEVICES"] = config.resources.policy_cuda
        env3["XDG_CACHE_HOME"] = config.resources.cache_dir
        try:
            policy_proc = subprocess.Popen(["accelerate", "launch", "engine_launcher.py"] + args_to_forward,
                                       env=env3, start_new_session=True)
            policy_proc.wait()
        except KeyboardInterrupt:
            # Send SIGINT to the entire process group
            os.killpg(os.getpgid(policy_proc.pid), signal.SIGINT)

            # Optional: Force kill if still running
            try:
                policy_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(policy_proc.pid), signal.SIGKILL)

    elif args.ctrain:
        pass
    elif args.test: # NOTE: I am just using the exact same implementation as training for now... may be better to change otherwise can merge this with above. 
        util.info('main.py', 'Preparing for Evaluation...')
        env1 = os.environ.copy()
        # deploy vLLM for TRL action sampling, use by "python main.py"
        env1["VLLM_USE_V1"] = "0"
        env1["CUDA_VISIBLE_DEVICES"] = config.resources.action_cuda
        num_gpus = len(config.resources.action_cuda.split(","))
        env1["XDG_CACHE_HOME"] = config.resources.cache_dir

        if args.compute_capability is not None:
            if args.compute_capability[0] >= 8:  # for GPUs with compute capability >= 8.0, can use bfloat16
                dtype = "bfloat16"
            else:  # for older GPUs, must use float16
                dtype = "half"
        else:
            dtype = "half"  # default to half if no GPU is found
    
        # run trl vllm serve
        action_proc = subprocess.Popen(["trl",
                                        "vllm-serve",
                                        f"--model={config.model.policy}",
                                        f"--gpu_memory_utilization={config.resources.action_gpu_memory_utilization}",
                                        f"--tensor_parallel_size={num_gpus}",
                                        f"--host=localhost",
                                        f"--port={config.resources.action_port}",
                                        f"--dtype={dtype}"],
                                       env=env1,
                                       stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL,
                                       start_new_session=True)

        # deploy vLLM for downstream, use by "python main.py"
        downstream_proc = None
        if not args.is_closed_source_downstream:
            env2 = os.environ.copy()
            env2["CUDA_VISIBLE_DEVICES"] = config.resources.downstream_cuda
            num_gpus = len(config.resources.downstream_cuda.split(","))
            env2["XDG_CACHE_HOME"] = config.resources.cache_dir
            # run vllm serve
            downstream_proc = subprocess.Popen(["vllm",
                                                "serve",
                                                config.model.downstream,
                                                f"--gpu_memory_utilization={config.resources.downstream_gpu_memory_utilization}",
                                                f"--tensor_parallel_size={num_gpus}",
                                                f"--host=localhost",
                                                f"--allowed-local-media-path={config.dataset.image_root}",
                                                f"--port={config.resources.downstream_port}",
                                                f"--dtype={dtype}"],
                                               env=env2,
                                               stdout=subprocess.DEVNULL,
                                               stderr=subprocess.DEVNULL,
                                               start_new_session=True)

        # wait for vLLM to be ready
        if not args.is_closed_source_downstream:
            # check if two ports are available for querying
            command = [
                "vllm", "serve", config.model.downstream,
                f"--gpu_memory_utilization={config.resources.downstream_gpu_memory_utilization}",
                f"--tensor_parallel_size={num_gpus}",
                "--host=localhost",
                f"--port={config.resources.downstream_port}",
                f"--dtype={dtype}",
                f"--allowed-local-media-path={config.dataset.image_root}"
            ]
            util.info('main.py', f"Launching vLLM with command: {' '.join(command)}")
            util.info('main.py', 'Waiting for vLLM to be ready (for downstream)...')
            wait_until_ready(port=config.resources.downstream_port, subproc=downstream_proc)
        util.info('main.py', 'Waiting for TRL vLLM-Serve to be ready (for action sampling)...')
        wait_until_ready(port=config.resources.action_port, subproc=action_proc)
        util.info('main.py', 'vLLMs are ready!')

        # deploy TRL for policy update/training, use by "accelerate launch main.py"
        env3 = os.environ.copy()
        env3["CUDA_VISIBLE_DEVICES"] = config.resources.policy_cuda
        env3["XDG_CACHE_HOME"] = config.resources.cache_dir
        try:
            policy_proc = subprocess.Popen(["accelerate", "launch", "engine_launcher.py"] + args_to_forward,
                                       env=env3, start_new_session=True)
            policy_proc.wait()
        except KeyboardInterrupt:
            # Send SIGINT to the entire process group
            os.killpg(os.getpgid(policy_proc.pid), signal.SIGINT)

            # Optional: Force kill if still running
            try:
                policy_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(policy_proc.pid), signal.SIGKILL)



if __name__ == "__main__":
    sys.exit(main())
