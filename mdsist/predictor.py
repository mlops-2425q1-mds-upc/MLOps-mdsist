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

class Predictor:
    """
    """

    def __init__(
        self,
        model: nn.Module,
        device: Union[str, torch.device, None] = None,
    ) -> None:
        """Initializes the Trainer.

        Args:
            model (nn.Module): The neural network model to be trained.
            optimizer (optim.Optimizer): The optimizer used for training.
            device (str | torch.device | None): The device to run the model on (CPU/GPU).
        """
        self.model = model
        self.device = device
        if self.device is None:
            self.device = util.get_available_device()

# Define ValueLoader so it take the input image, convert it into 28x28 matrix, in uint8

    def predict(self, images):
        """
        Args:

        Returns:
            
        """
        self.model.eval()
        val_preds = []
        with torch.no_grad():# no recalcula el model 
            for image in images:
                image = image.to(self.device)
                #labels = labels.to(self.device)
                outputs = self.model(image)  # input as numpy 28x28 matrix. uint8
                preds = outputs.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                return preds