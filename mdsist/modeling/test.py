"""
Module for evaluating a machine learning model using data from a test set and logging results to
MLflow.

This script provides a command-line interface (CLI) using `typer` to evaluate a trained model
stored in MLflow. The test dataset path is provided as an argument, and the model is evaluated
on various metrics such as accuracy, F1 score, precision, recall, and loss. These metrics are
logged to MLflow for tracking and comparison.

The module includes:
- Data loading and preprocessing using PyTorch.
- Device selection (GPU/CPU).
- Integration with MLflow to retrieve a stored model and log evaluation metrics.
- A custom exception to handle cases where the model is not found.

Dependencies:
- PyTorch, torchvision, and related utilities.
- MLflow for experiment tracking and logging.
- Typer for creating the CLI interface.
- Loguru for logging.
"""

import os
from pathlib import Path
from typing import Union

import mlflow
import mlflow.artifacts
import mlflow.pytorch
import torch
import typer
import yaml
from dotenv import load_dotenv
from loguru import logger
from torch.utils.data import DataLoader
from torchvision import transforms

from mdsist import util
from mdsist.dataset import MdsistDataset
from mdsist.trainer import Trainer

app = typer.Typer()


class ModelNotFound(Exception):
    """
    Raise when Model is not found in MLFLow
    """

    def __init__(self):
        super().__init__("Logged model not found!")


@app.command()
def main(
    test_set_path: Path,
    device: Union[str, None] = None,
) -> None:
    """
    Main function to evaluate a trained model on a test dataset.

    Args:
        test_set_path (Path): Path to the test dataset to be evaluated.
        device (Union[str, None], optional): The device on which to run the evaluation.
            If `None`, the best available device (GPU or CPU) is automatically selected.

    Raises:
        ModelNotFound: If the model cannot be found in the MLflow artifacts for the current run.

    Returns:
        None
    """
    with open("params.yaml", encoding="utf-8") as param_file:
        params = yaml.safe_load(param_file)["testing"]

    experiment_id = params["experiment_id"]
    batch_size = params["batch_size"]

    # Load environment variables
    load_dotenv()

    # Define the transformation to normalize images
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Load datasets
    test_dataset = MdsistDataset(test_set_path, transform=transform)

    # Create DataLoaders
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Set device
    if device is None:
        device = util.get_available_device()
    else:
        device = torch.device(device)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(experiment_id)

    # Get the current run id
    current_run = mlflow.search_runs(max_results=1).iloc[0]

    artifacts = mlflow.artifacts.list_artifacts(current_run.artifact_uri)
    model_uri = None
    for artifact in artifacts:
        if "model" in artifact.path:
            model_uri = f'runs:/{current_run.run_id}/{artifact.path.split("/")[-1]}'
            break

    if model_uri is None:
        raise ModelNotFound()

    # Instantiate model
    model = mlflow.pytorch.load_model(model_uri)
    model.to(device)

    # We can leverage Trainer 'validate' method
    evaluator = Trainer(model, device=device)
    test_stats = evaluator.validate(test_loader)

    logger.info(test_stats)

    with mlflow.start_run(current_run.run_id):

        # Log hyperparameters
        mlflow.log_metric("test_accuracy", test_stats.accuracy)
        mlflow.log_metric("test_f1_score", test_stats.f1_score)
        mlflow.log_metric("test_precision", test_stats.precision)
        mlflow.log_metric("test_recall", test_stats.recall)
        mlflow.log_metric("test_loss", test_stats.loss)


if __name__ == "__main__":
    app()
