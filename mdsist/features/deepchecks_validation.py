"""
Deepchecks Validation

This script first obtains the train and test datasets as DataLoader structures 
and then performs the Deepchecks validation.
"""

from deepchecks.core import CheckResult
from deepchecks.vision import SingleDatasetCheck, VisionData
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


class ImageDimensionCheck(SingleDatasetCheck):
    """
    Class ImageDimensionCheck

    Custom check to validate image dimensions
    """

    def __init__(self):
        super().__init__(name="Image Dimension Check")
        self.incorrect_dims = []
        self.expected_shape = (28, 28, 1)

    def params(self, show_defaults=False):
        return {}

    def metadata(self, with_doc_link=False):
        return {
            "summary": "Checks if the dimensions of the images are the expected ones, 28x28",
            "doc_link": None,
        }

    def initialize_run(self, context, dataset_kind):
        self.incorrect_dims = []

    def update(self, context, batch, dataset_kind):
        images = batch.original_images
        for img in images:
            if img.shape != self.expected_shape:
                self.incorrect_dims.append(img.shape)

    def compute(self, context, dataset_kind):
        result_value = {"incorrect_dimensions": self.incorrect_dims}

        if context.with_display:
            # Generate a simple HTML display showing incorrect image dimensions
            display_html = ""
            if self.incorrect_dims:
                display_html += "<h4>Incorrect Image Dimensions</h4>"
                display_html += "<ul>"
                for dim in set(self.incorrect_dims):
                    display_html += f"<li>Found image with dimensions: {dim}</li>"
                display_html += "</ul>"
            else:
                display_html += "<p>All images have correct dimensions.</p>"

            return CheckResult(result_value, display=display_html)

        return CheckResult(result_value, display=None)


custom_suite = full_suite()

# Remove checks that do not apply to our dataset
custom_suite.remove(0)
custom_suite.remove(1)
custom_suite.remove(2)
custom_suite.remove(3)
custom_suite.remove(4)
custom_suite.remove(5)
custom_suite.remove(6)
custom_suite.remove(14)

# Add the custom image dimension check
custom_suite.add(ImageDimensionCheck())

result = custom_suite.run(train_ds, test_ds)

OUTPUT_FILE = REPORTS_DIR / "deepchecks_validation.html"

# If the output file already exists, delete it to avoid duplicates
if OUTPUT_FILE.exists():
    OUTPUT_FILE.unlink()

result.save_as_html(str(OUTPUT_FILE))
