"""
This module defines a Convolutional Neural Network (CNN) model using PyTorch.
It consists of two convolutional layers followed by two fully connected layers, with max 
pooling applied after each convolutional layer.
"""

from torch import nn
from torch.nn import functional as F


class CNN(nn.Module):
    """Convolutional Neural Network (CNN) model for image classification.

    This model consists of two convolutional layers followed by two fully connected
    layers. It is designed to process grayscale images and classify them into 10 classes.
    """

    def __init__(self):
        """Initializes the CNN architecture.

        This method sets up the layers of the CNN, including convolutional layers,
        fully connected layers, and a max pooling layer.
        """
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """Defines the forward pass of the CNN.

        Args:
            x (torch.Tensor): Input tensor of shape (N, 1, H, W), where N is the batch size,
                              H is the height, and W is the width of the image.

        Returns:
            torch.Tensor: Output tensor of shape (N, 10), representing the class scores
                          for each input image.
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
