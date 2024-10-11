import os
import torch
import subprocess


def set_env() -> None:
    os.environ['REQUESTS_CA_BUNDLE'] = './certs/concatenated SSL bundle cert.cert'

    # prevent SSL connection Proxy/timeout errors with HuggingFace Datasets and Models
    terminal_command = "pip install --trusted-host=pypi.org --trusted-host=files.pythonhosted.org pip-system-certs'"

    # Execute the terminal command within the script
    subprocess.run(terminal_command, shell=True)

def det_gpu_status() -> str:
    # check if CUDA is available
    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')
        torch.cuda.empty_cache()
        print(f'Memory allocated: {torch.cuda.memory_allocated()}')
        print(f'Max memory allocated: {torch.cuda.max_memory_allocated()}')

    return train_on_gpu

if __name__ == "__main__":
    set_env()