"""
Unit tests for utility functions in the mdsist project.
"""

import random

import numpy as np
import pytest
import torch
from loguru import logger

# Import the utility functions
from mdsist.util import (
    get_available_device,
    get_current_timestamp_formatted,
    log_model_complexity,
    seed_all,
)


def test_get_current_timestamp_formatted():
    """
    Test the get_current_timestamp_formatted function to ensure it returns
    a timestamp in the expected format (YYYYmmDDhhmmss).
    """
    timestamp = get_current_timestamp_formatted()
    assert len(timestamp) == 14, "Timestamp should be in YYYYmmDDhhmmss format."
    assert timestamp.isdigit(), "Timestamp should only contain digits."


def test_seed_all():
    """
    Test the seed_all function to ensure it properly seeds all random generators
    (numpy, Python, and PyTorch) for reproducibility.
    """
    seed = 42
    seed_all(seed)

    # Check if seeding produces the same values in numpy
    np_value_1 = np.random.rand(1)
    seed_all(seed)
    np_value_2 = np.random.rand(1)
    assert np.array_equal(
        np_value_1, np_value_2
    ), "Numpy random values should be equal after re-seeding."

    # Check if seeding produces the same values in Python's random
    random_value_1 = random.random()
    seed_all(seed)
    random_value_2 = random.random()
    assert (
        random_value_1 == random_value_2
    ), "Python random values should be equal after re-seeding."

    # Check if seeding produces the same values in PyTorch
    torch_value_1 = torch.rand(1)
    seed_all(seed)
    torch_value_2 = torch.rand(1)
    assert torch.equal(
        torch_value_1, torch_value_2
    ), "PyTorch random values should be equal after re-seeding."


def test_get_available_device():
    """
    Test the get_available_device function to ensure it returns a valid
    torch.device ('cpu' or 'cuda').
    """
    device = get_available_device()
    assert isinstance(device, torch.device), "The returned device should be of type torch.device."
    assert device.type in ["cpu", "cuda"], "Device should be either 'cpu' or 'cuda'."


def test_log_model_complexity(monkeypatch):
    """
    Test the log_model_complexity function to ensure it logs the number of
    FLOPS and parameters for a given model.
    """

    class SimpleModel(torch.nn.Module):
        """A simple model for testing log_model_complexity."""

        def __init__(self):
            super().__init__()  # Use Python 3 style super()
            self.fc = torch.nn.Linear(28 * 28, 10)

        def forward(self, x):
            """Forward pass for the model."""
            return self.fc(x)

    # Mock the logger to capture log messages
    log_messages = []

    def mock_info(msg):
        log_messages.append(msg)

    monkeypatch.setattr(logger, "info", mock_info)

    # Create a simple model
    model = SimpleModel()

    # Call the function
    log_model_complexity(model)

    # Check if FLOPS and Parameters are logged
    assert any("FLOPS" in msg for msg in log_messages), "FLOPS should be logged."
    assert any("Parameters" in msg for msg in log_messages), "Parameters should be logged."


if __name__ == "__main__":
    pytest.main()
