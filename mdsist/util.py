import datetime as dt
import random

import numpy as np
import ptflops
import torch
from loguru import logger


def get_current_YYYYmmDDhhmmss() -> str:
    """
    Get the current timestamp in YYYYmmDDhhmmss format

    Returns
    -------
    str
        The current timestamp in YYYYmmDDhhmmss format
    """

    return dt.datetime.now().strftime("%Y%m%d%H%M%S")


def seed_all(value: int) -> None:
    """
    Seed all random number generators for reproducibility

    Parameters
    ----------
    value : int
        The seed value to use for random number generators
    """

    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)  # Set seed for CUDA
    torch.cuda.manual_seed_all(value)  # Use this to ensure that all GPUs get the same seed
    random.seed(value)

    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior for cudnn
    torch.backends.cudnn.benchmark = False  # Disable cudnn's benchmark mode for reproducibility


def get_available_device() -> torch.device:
    """
    Get the available device for PyTorch

    Returns
    -------
    torch.device
        The available device for PyTorch
    """

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log_model_complexity(model: torch.nn.Module) -> None:
    """
    Log the model complexity: number of parameters and number of FLOPS.

    Parameters
    ----------
    model : torch.nn.Module
        The model to analize
    """

    model.eval()

    flops, params = ptflops.get_model_complexity_info(
        model, (1, 28, 28), print_per_layer_stat=False
    )

    logger.info(f"FLOPS: {flops}")
    logger.info(f"Parameters: {params}")
