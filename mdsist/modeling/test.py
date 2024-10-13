import os
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
import torchvision.transforms as transforms
import typer
import yaml
from dotenv import load_dotenv
from loguru import logger
from torch.utils.data import DataLoader

import mdsist.util as util
from mdsist.dataset import MdsistDataset
from mdsist.trainer import Trainer

app = typer.Typer()


@app.command()
def main(
    test_set_path: Path,
    model_uri: Path,
    device: str | None = None,
) -> None:

    params = yaml.safe_load(open("params.yaml"))["training"]

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

    # Instantiate model
    model = mlflow.pytorch.load_model(model_uri)
    model.to(device)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(experiment_id)

    # We can leverage Trainer 'validate' method

    evaluator = Trainer(model, device=device)
    test_stats = evaluator.validate(test_loader)

    logger.info(test_stats)

    with mlflow.start_run():
        mlflow.set_tag("mlflow.runName", "Test")

        # Log hyperparameters
        mlflow.log_metric("test_accuracy", test_stats.accuracy)
        mlflow.log_metric("test_f1_score", test_stats.f1_score)
        mlflow.log_metric("test_precision", test_stats.precision)
        mlflow.log_metric("test_recall", test_stats.recall)
        mlflow.log_metric("test_loss", test_stats.loss)


if __name__ == "__main__":
    app()
