import torch
from typing import Union


def get_device(ignore_mps: bool = False, cuda_as_int: bool = False) -> Union[str, int]:
    """
    Determines and returns a PyTorch device name.

    :param ignore_mps: Manual deactivate MPS
    :param cuda_as_int: Return cuda as device number
    :return: device name as str
    """
    # Define device (either GPU, M1/2, or CPU)
    if torch.cuda.is_available():
        print('Device: using CUDA')
        return "cuda" if not cuda_as_int else 0
    elif torch.backends.mps.is_available() and not ignore_mps:
        print('Device: using MPS')
        return "mps"
    else:
        print('Device: using CPU :(')
        return "cpu"
