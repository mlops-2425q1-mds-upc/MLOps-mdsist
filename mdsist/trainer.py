from dataclasses import dataclass

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

import mdsist.util as util


@dataclass
class Stats:
    accuracy: float
    precision: float
    recall: float
    f1_score: float


@dataclass
class TrainStats(Stats):
    loss: float


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        device: str | torch.device | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_function = nn.CrossEntropyLoss()
        self.device = device
        if self.device is None:
            self.device = util.get_available_device()

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> None:
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

            logger.info(f"Epoch [{epoch+1}/{epochs}]")
            logger.info(
                f"[Train] Loss: {train_stats.loss:.4f} | Accuracy: {train_stats.accuracy:.4f} | "
                f"Precision: {train_stats.precision:.4f} | Recall: {train_stats.recall:.4f} | F1 Score: {train_stats.f1_score:.4f}"
            )
            logger.info(
                f"[Val  ] Loss: {val_stats.loss:.4f} | Accuracy: {val_stats.accuracy:.4f} | "
                f"Precision: {val_stats.precision:.4f} | Recall: {val_stats.recall:.4f} | F1 Score: {val_stats.f1_score:.4f}"
            )

        logger.info("Training completed.")

    def train_epoch(self, train_loader: DataLoader) -> TrainStats:
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
