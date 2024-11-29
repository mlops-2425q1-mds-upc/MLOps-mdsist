"""
Tests for the Test module in the mdsist project.
"""

import os
from pathlib import Path

import mlflow
import pytest
import torch
import yaml
from dotenv import load_dotenv
from mlflow.exceptions import MlflowException
from torch.utils.data import DataLoader
from torchvision import transforms

from mdsist.dataset import MdsistDataset
from mdsist.trainer import Trainer


class ModelNotFound(Exception):
    """
    Raised when the model is not found in MLflow.
    """

    def __init__(self, experiment_id):
        message = f"Logged model not found for experiment ID: {experiment_id}!"
        super().__init__(message)


@pytest.fixture
def mnist_model():
    """
    Fixture to load the saved PyTorch model from MLflow.
    """
    with open("params.yaml", encoding="utf-8") as param_file:
        params = yaml.safe_load(param_file)["testing"]

    experiment_id = params["experiment_id"]

    # Load environment variables
    load_dotenv()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(experiment_id)

    try:
        # Get the latest run in the experiment
        runs = mlflow.search_runs(max_results=1)
        if runs.empty:
            raise ModelNotFound(experiment_id)

        current_run = runs.iloc[0]

        # Fetch artifacts to find the model
        artifacts = mlflow.artifacts.list_artifacts(current_run.artifact_uri)
        model_uri = None
        for artifact in artifacts:
            if "model" in artifact.path:
                model_uri = f'runs:/{current_run.run_id}/{artifact.path.split("/")[-1]}'
                break

        if model_uri is None:
            raise ModelNotFound(experiment_id)

        # Load and return the model
        model = mlflow.pytorch.load_model(model_uri)
        return model

    except MlflowException as e:
        raise RuntimeError(f"Error while loading the model from MLflow: {e}") from e


@pytest.fixture
def test_data_loader():
    """
    Fixture to create mock DataLoaders for testing using the test MNIST dataset.
    """
    with open("params.yaml", encoding="utf-8") as param_file:
        params = yaml.safe_load(param_file)["testing"]

    batch_size = params["batch_size"]

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]  # Normalize dataset
    )

    root_dir = Path(__file__).parents[2]
    path_to_test = os.path.join(root_dir, "data/processed/test.parquet")

    test_dataset = MdsistDataset(path_to_test, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader


@pytest.fixture
def test_threshold():
    """
    Fixture to define the minimum threshold our model should pass
    """

    # pylint: disable=R0903
    class TestStats:
        """
        Test threshold class
        """

        def __init__(self):
            self.accuracy = 0.9  # Threshold for accuracy

    # pylint: enable=R0903
    return TestStats()


def test_data(test_data_loader):
    """
    Test data loader is correctly defined
    """
    assert len(test_data_loader.dataset) > 0, "The test dataset is empty!"

    assert (
        test_data_loader.batch_size == 64
    ), f"Expected batch size of 64, but got {test_data_loader.batch_size}"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mnist_model(mnist_model, test_data_loader, test_threshold, device):
    """
    Test we get the minimum accuracy required
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    evaluator = Trainer(mnist_model, device=device)

    # Simulate the evaluation process
    test_stats = evaluator.validate(test_data_loader)

    # Using test_threshold to compare the accuracy
    assert (
        test_stats.accuracy > test_threshold.accuracy
    ), f"Expected accuracy > {test_threshold.accuracy}, but got {test_stats.accuracy}"
