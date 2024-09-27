from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose


class MdsistDataset(Dataset):
    def __init__(self, file_path: Path, transform: Compose | None = None):
        self.data = pd.read_parquet(file_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data.iloc[idx, :-1].values.astype("float32")
        label = int(self.data.iloc[idx, -1])
        image = image.reshape(1, 28, 28)

        if self.transform:
            image = self.transform(image)

        return image, label
