import os
from pathlib import Path

import codecarbon as cc
import mlflow
import mlflow.pytorch
import torch
import torchvision.transforms as transforms
import typer
from dotenv import load_dotenv
from loguru import logger
from torch.utils.data import DataLoader
import yaml
import sys

import mdsist.util as util
from mdsist.architectures import CNN
from mdsist.config import MODELS_DIR, PROCESSED_DATA_DIR
from mdsist.dataset import MdsistDataset
from mdsist.trainer import Trainer

app = typer.Typer()


@app.command()
def main(
    train_set_path: Path,
    val_set_path: Path,
    model_path: Path,
    device: str | None = None,
) -> None:
    
    params = yaml.safe_load(open("params.yaml"))["training"]
    
    experiment_id = params["experiment_id"]
    seed = params['seed']
    epochs = params["epochs"]
    batch_size = params["batch_size"]
    learning_rate = params['learning_rate']

    # Load environment variables
    load_dotenv()

    # Set seed for reproducibility
    util.seed_all(seed)

    # Define the transformation to normalize images
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Load datasets
    train_dataset = MdsistDataset(train_set_path, transform=transform)
    val_dataset = MdsistDataset(val_set_path, transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Set device
    if device is None:
        device = util.get_available_device()
    else:
        device = torch.device(device)

    # Instantiate model
    model = CNN()
    model.to(device)

    # Log model complexity (params and flops)
    util.log_model_complexity(model)

    # Get optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(experiment_id)

    with mlflow.start_run():
        mlflow.set_tag("mlflow.runName", "Train")

        # Log hyperparameters
        mlflow.log_param("seed", seed)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)

        # Start emissions tracking
        emissions_tracker = cc.EmissionsTracker(project_name="MDSIST", experiment_id=experiment_id)
        emissions_tracker.start()

        # Instantiate a trainer that will handle the training process
        trainer = Trainer(model, optimizer, device)

        # Train
        trainer.train(train_loader, val_loader, epochs)

        # Stop emissions tracking
        emissions = emissions_tracker.stop()

        # Log emissions
        mlflow.log_metric("emissions_kg_co2", emissions)

        # Log the model to MLFlow
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        mlflow.pytorch.log_model(trainer.model, model_name)

        # Save the model
        torch.save(trainer.model, model_path)

        logger.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    app()
