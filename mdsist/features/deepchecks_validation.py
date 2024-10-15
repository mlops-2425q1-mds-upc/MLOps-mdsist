from mdsist.config import JPG_IMAGES_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from mdsist.features.parser import ParquetJPGParser
import os
from deepchecks.vision import classification_dataset_from_directory, VisionData
from deepchecks.vision.suites import data_integrity, train_test_validation
from mdsist.dataset import MdsistDataset
from torch.utils.data import DataLoader
from torchvision import transforms

"""
Deepchecks Validation

This script first parses (if needed) the processed data from parquet to jpg. Then, performs the validation of the data
using Deepchecks.
"""
"""
# check if train and test folders in images_jpg exist, if not, parse train and test processed parquet files to jpg images
if not os.path.isdir(JPG_IMAGES_DIR / 'train'):
    train_jpg = ParquetJPGParser(PROCESSED_DATA_DIR / "train.parquet", JPG_IMAGES_DIR / 'train')
    train_jpg.save_images()

if not os.path.isdir(JPG_IMAGES_DIR / 'test'):
    test_jpg = ParquetJPGParser(PROCESSED_DATA_DIR / "test.parquet", JPG_IMAGES_DIR / 'test')
    test_jpg.save_images()

# deepchecks validation
train_ds, test_ds = classification_dataset_from_directory(JPG_IMAGES_DIR, object_type="VisionData")
"""

# Define the transformation to normalize images
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load datasets
train_dataset = MdsistDataset(PROCESSED_DATA_DIR / "train.parquet", transform=transform, return_dict=True)
test_dataset = MdsistDataset(PROCESSED_DATA_DIR / "test.parquet", transform=transform, return_dict = True)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

train_ds = VisionData(batch_loader=train_loader, task_type="classification", dataset_name="MDSIST", reshuffle_data=True)
test_ds = VisionData(batch_loader=test_loader, task_type="classification", dataset_name="MDSIST", reshuffle_data=True)
custom_suite = data_integrity()

custom_suite.add(train_test_validation())

result = custom_suite.run(train_ds, test_ds)

OUTPUT_FILE = REPORTS_DIR / "deepchecks_validation.html"

# If the output file already exists, delete it to avoid duplicates
if OUTPUT_FILE.exists():
    OUTPUT_FILE.unlink()

result.save_as_html(str(OUTPUT_FILE))