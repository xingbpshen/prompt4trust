from huggingface_hub import repo_exists
import os


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
