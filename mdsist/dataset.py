import io
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose


class MdsistDataset(Dataset):
    def __init__(self, file_path: Path, transform: Compose | None = None):
        self.data = pd.read_parquet(file_path, engine="pyarrow")
        self.data = self.data.reset_index(drop=True)
        self.transform = transform

    def decode_png_image(self, image_dict):
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
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve the image dictionary and label
        image_dict = self.data.loc[idx, "image"]
        label = self.data.loc[idx, "label"]

        # Decode the PNG image
        image_array = self.decode_png_image(image_dict)

        image = image_array.reshape(1, 28, 28)

        if self.transform:
            image = self.transform(image)

        image = image.reshape(1, 28, 28)

        return image, label
