"""
Trainer classes for handling model training and validation.
"""

from typing import Union

import torch
from torch import nn

from mdsist import util


class Predictor:
    """Predictor class"""

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

    def get_model(self):
        """Get predictor model"""
        return self.model

    # Define ValueLoader so it take the input image, convert it into 28x28 matrix, in uint8

    def predict(self, images):
        """
        Args:

        Returns:

        """
        self.model.eval()
        val_preds = []
        with torch.no_grad():  # no recalcula el model
            for image in images:
                image = image.to(self.device)
                outputs = self.model(image.float().clone())  # input as numpy 28x28 matrix. uint8
                preds = outputs.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
            return val_preds
