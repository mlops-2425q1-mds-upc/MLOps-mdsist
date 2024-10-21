"""
Train a Convolutional Neural Network (CNN) model on a specified training dataset and 
validate it using a validation dataset.

This script utilizes the following libraries:
- PyTorch for building and training the model
- MLflow for tracking experiments and logging metrics
- CodeCarbon for tracking carbon emissions during training
- Typer for creating command-line interfaces
- dotenv for loading environment variables
- Loguru for logging

Usage:
    Run the script from the command line with the required parameters:
    
    python script_name.py --train-set-path <path_to_train_set> \
                           --val-set-path <path_to_val_set> \
                           --model-path <path_to_save_model> \
                           [--device <device_name>]

Parameters:
    train_set_path (Path): Path to the training dataset.
    val_set_path (Path): Path to the validation dataset.
    model_path (Path): Path to save the trained model.
    device (str | None): Optional; specify the device to train on (e.g., 'cuda' or 'cpu'). 
    If not provided, the script will automatically choose the available device.

The script loads parameters from a `params.yaml` configuration file and environment variables from 
a `.env` file. It sets a random seed for reproducibility, normalizes the images,
and creates DataLoaders for the training and validation datasets.

The training process is managed by the `Trainer` class, which handles the training loop. 
During training, the script tracks carbon emissions and logs various metrics, including the model's 
hyperparameters and emissions, to MLflow. After training, the model is saved to the specified path.

"""

import os
from pathlib import Path
from typing import Union

import codecarbon as cc
import mlflow
import mlflow.pytorch
import torch
import typer
import yaml
from dotenv import load_dotenv
from loguru import logger
from torch.utils.data import DataLoader
from torchvision import transforms

from mdsist import util
from mdsist.architectures import CNN
from mdsist.dataset import MdsistDataset
from mdsist.trainer import Trainer

app = typer.Typer()


@app.command()
def main(
    train_set_path: Path,
    val_set_path: Path,
    model_path: Path,
    device: Union[str, None] = None,
) -> None:
    """Train a CNN model on the provided training and validation datasets.

    Args:
        train_set_path (Path): Path to the training dataset.
        val_set_path (Path): Path to the validation dataset.
        model_path (Path): Path to save the trained model.
        device (str | None): Optional; specify the device to use (e.g., 'cuda' or 'cpu').
                             If None, automatically selects the available device.

    This function loads training parameters from a YAML file, prepares the datasets and
    data loaders, and then trains the CNN model while tracking emissions and logging
    metrics with MLflow.
    """

    with open("params.yaml", encoding="utf-8") as param_file:
        params = yaml.safe_load(param_file)["training"]

    experiment_id = params["experiment_id"]
    seed = params["seed"]
    epochs = params["epochs"]
    batch_size = params["batch_size"]
    learning_rate = params["learning_rate"]

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
