import torch


def get_device(ignore_mps: bool = False) -> str:
    """
    Determines and returns a PyTorch device name.

    :param ignore_mps: Manual deactivate MPS
    :return: device name as str
    """
    # Define device (either GPU, M1/2, or CPU)
    if torch.cuda.is_available():
        print('Device: using CUDA')
        return "cuda"
    elif torch.backends.mps.is_available() and not ignore_mps:
        print('Device: using MPS')
        return "mps"
    else:
        print('Device: using CPU :(')
        return "cpu"
