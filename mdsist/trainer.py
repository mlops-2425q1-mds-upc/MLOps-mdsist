"""
Trainer classes for handling model training and validation.
"""

from dataclasses import dataclass
from typing import Union

import mlflow
import torch
import tqdm
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn, optim
from torch.utils.data import DataLoader

from mdsist import util


@dataclass
class Stats:
    """Stats obtained from the model training"""

    accuracy: float
    precision: float
    recall: float
    f1_score: float


@dataclass
class TrainStats(Stats):
    """Stats required for the model training"""

    loss: float


class Trainer:
    """Trainer class for handling model training and validation.

    This class encapsulates the training and validation processes for a PyTorch model,
    logging relevant metrics to MLflow.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Union[optim.Optimizer, None] = None,
        device: Union[str, torch.device, None] = None,
    ) -> None:
        """Initializes the Trainer.

        Args:
            model (nn.Module): The neural network model to be trained.
            optimizer (optim.Optimizer): The optimizer used for training.
            device (str | torch.device | None): The device to run the model on (CPU/GPU).
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_function = nn.CrossEntropyLoss()
        self.device = device
        if self.device is None:
            self.device = util.get_available_device()

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> None:
        """Trains the model for a specified number of epochs.

        Args:
            train_loader (DataLoader): DataLoader for the training set.
            val_loader (DataLoader): DataLoader for the validation set.
            epochs (int): Number of epochs to train the model.
        """
        self.model.to(self.device)

        logger.info(f"Start training for {epochs} epochs...")
        for epoch in tqdm.tqdm(range(epochs), total=epochs):
            train_stats = self.train_epoch(train_loader)
            val_stats = self.validate(val_loader)

            # Log train metrics
            mlflow.log_metric("train_accuracy", train_stats.accuracy, epoch)
            mlflow.log_metric("train_precision", train_stats.precision, epoch)
            mlflow.log_metric("train_recall", train_stats.recall, epoch)
            mlflow.log_metric("train_f1_score", train_stats.f1_score, epoch)

            # Log val metrics
            mlflow.log_metric("val_accuracy", val_stats.accuracy, epoch)
            mlflow.log_metric("val_precision", val_stats.precision, epoch)
            mlflow.log_metric("val_recall", val_stats.recall, epoch)
            mlflow.log_metric("val_f1_score", val_stats.f1_score, epoch)

            logger.info(f"Epoch [{epoch + 1}/{epochs}]")
            logger.info(
                f"[Train] Loss: {train_stats.loss:.4f} | Accuracy: {train_stats.accuracy:.4f} | "
                f"Precision: {train_stats.precision:.4f} | Recall: {train_stats.recall:.4f}"
                + f" | F1 Score: {train_stats.f1_score:.4f}"
            )
            logger.info(
                f"[Val  ] Loss: {val_stats.loss:.4f} | Accuracy: {val_stats.accuracy:.4f} | "
                f"Precision: {val_stats.precision:.4f} | Recall: {val_stats.recall:.4f}"
                + f" | F1 Score: {val_stats.f1_score:.4f}"
            )

        logger.info("Training completed.")

    def train_epoch(self, train_loader: DataLoader) -> TrainStats:
        """Conducts one training epoch.

        Args:
            train_loader (DataLoader): DataLoader for the training set.

        Returns:
            TrainStats: A dataclass containing training metrics including accuracy, precision,
            recall, F1 score, and loss for the epoch.
        """
        # Training phase
        self.model.train()
        loss = 0
        train_preds, train_labels = [], []

        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(images)
            batch_loss = self.loss_function(outputs, labels)
            batch_loss.backward()
            self.optimizer.step()

            loss += batch_loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().numpy())

        loss /= len(train_loader)
        accuracy = accuracy_score(train_labels, train_preds)
        precision = precision_score(train_labels, train_preds, average="macro")
        recall = recall_score(train_labels, train_preds, average="macro")
        f1_scr = f1_score(train_labels, train_preds, average="macro")

        return TrainStats(accuracy, precision, recall, f1_scr, loss)

    def validate(self, val_loader: DataLoader) -> TrainStats:
        """Evaluates the model on the validation set.

        Args:
            val_loader (DataLoader): DataLoader for the validation set.

        Returns:
            TrainStats: A dataclass containing validation metrics including accuracy, precision,
            recall, F1 score, and loss for the epoch.
        """
        self.model.eval()
        loss = 0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                batch_loss = self.loss_function(outputs, labels)
                loss += batch_loss.item()

                preds = outputs.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())

        loss /= len(val_loader)
        accuracy = accuracy_score(val_labels, val_preds)
        precision = precision_score(val_labels, val_preds, average="macro")
        recall = recall_score(val_labels, val_preds, average="macro")
        f1_scr = f1_score(val_labels, val_preds, average="macro")

        return TrainStats(accuracy, precision, recall, f1_scr, loss)
