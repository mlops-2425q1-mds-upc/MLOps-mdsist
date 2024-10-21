"""
Deepchecks Validation

This script first obtains the train and test datasets as DataLoader structures 
and then performs the Deepchecks validation.
"""

from deepchecks.vision import VisionData
from deepchecks.vision.suites import full_suite
from torch.utils.data import DataLoader

from mdsist.config import PROCESSED_DATA_DIR, REPORTS_DIR
from mdsist.dataset import MdsistDataset

# Load datasets
train_dataset = MdsistDataset(PROCESSED_DATA_DIR / "train.parquet", deepchecks_format=True)
test_dataset = MdsistDataset(PROCESSED_DATA_DIR / "test.parquet", deepchecks_format=True)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

train_ds = VisionData(
    batch_loader=train_loader,
    task_type="classification",
    dataset_name="MDSIST_train",
    reshuffle_data=True,
)
test_ds = VisionData(
    batch_loader=test_loader,
    task_type="classification",
    dataset_name="MDSIST_test",
    reshuffle_data=True,
)
custom_suite = full_suite()

# remove checks that do not apply to our dataset

custom_suite.remove(0)
custom_suite.remove(1)
custom_suite.remove(2)
custom_suite.remove(3)
custom_suite.remove(4)
custom_suite.remove(5)
custom_suite.remove(6)
custom_suite.remove(14)

result = custom_suite.run(train_ds, test_ds)

OUTPUT_FILE = REPORTS_DIR / "deepchecks_validation.html"

# If the output file already exists, delete it to avoid duplicates
if OUTPUT_FILE.exists():
    OUTPUT_FILE.unlink()

result.save_as_html(str(OUTPUT_FILE))
