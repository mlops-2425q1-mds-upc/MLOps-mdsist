"""
Custom dataset class for loading images and labels from a parquet file.
"""

import io
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose


class MdsistDataset(Dataset):
    """Custom dataset class for loading images and labels from a parquet file."""

    def __init__(
        self,
        file_path: Path,
        transform: Union[Compose, None] = None,
        deepchecks_format: bool = False,
    ):
        """Initializes the dataset.

        Args:
            file_path (Path): Path to the parquet file containing the dataset.
            transform (Compose | None): Optional; a torchvision transform
            to be applied to the images.
            deepchecks_format (bool): If True, get the data formatted for deepchecks;
            if False, get a (image, label) tuple.
        """
        self.data = pd.read_parquet(file_path, engine="pyarrow")
        self.data = self.data.reset_index(drop=True)
        self.transform = transform
        self.deepchecks_format = deepchecks_format

    def decode_png_image(self, image_dict):
        """Decodes a PNG image from binary data.

        Args:
            image_dict (dict): A dictionary containing the binary PNG data under the key "bytes".

        Returns:
            np.ndarray: The decoded image as a NumPy array.
        """
        # Extract the binary png data
        png_bytes = image_dict.get("bytes")

        # Create a BytesIO object from the binary data
        image_stream = io.BytesIO(png_bytes)

        # Open the image using Pillow
        with Image.open(image_stream) as img:
            img = img.convert("L")
            # Convert the image to a NumPy array
            image_array = np.array(img)

        return image_array

    def __len__(self):
        """Returns the number of samples in the dataset.

        Returns:
            int: The total number of samples.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """Retrieves the image and label at the specified index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict or tuple: A dictionary (if deepchecks_format=True) containing images and labels,
            or a tuple (if deepchecks_format=False) containing (image, label).
        """
        image_dict = self.data.loc[idx, "image"]
        label = self.data.loc[idx, "label"]
        image_array = self.decode_png_image(image_dict)
        image = image_array.reshape((1, 28, 28))

        if self.transform:
            image = self.transform(image)

        # Deephecks requires a dictionary with channels last
        if self.deepchecks_format:
            image = image.reshape(28, 28, 1)
            return {"images": image, "labels": label}

        image = image_array.reshape((1, 28, 28))
        return image, label
