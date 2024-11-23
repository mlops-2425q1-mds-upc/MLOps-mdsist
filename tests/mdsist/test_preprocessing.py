import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml


@pytest.fixture
def sample_data():
    """Fixture to provide sample training and testing data."""
    train_data = pd.DataFrame(
        {
            "feature1": np.random.rand(10),
            "feature2": np.random.rand(10),
            "label": np.random.randint(0, 2, size=10),
        }
    )
    test_data = pd.DataFrame(
        {
            "feature1": np.random.rand(5),
            "feature2": np.random.rand(5),
            "label": np.random.randint(0, 2, size=5),
        }
    )
    return train_data, test_data


@pytest.fixture
def setup_environment(sample_data):
    """Fixture to set up the environment for testing, including params.yaml."""
    train_data, test_data = sample_data

    # Create temporary files for train and test data
    with (
        tempfile.NamedTemporaryFile(delete=False) as temp_train,
        tempfile.NamedTemporaryFile(delete=False) as temp_test,
        tempfile.NamedTemporaryFile(delete=False) as temp_processed_train,
        tempfile.NamedTemporaryFile(delete=False) as temp_processed_validation,
        tempfile.NamedTemporaryFile(delete=False) as temp_processed_test,
        tempfile.NamedTemporaryFile(delete=False, suffix=".yaml") as temp_params,
    ):

        train_path = temp_train.name
        test_path = temp_test.name
        processed_train_path = temp_processed_train.name
        processed_validation_path = temp_processed_validation.name
        processed_test_path = temp_processed_test.name
        params_path = temp_params.name

        # Save the sample data to parquet files
        pq.write_table(pa.Table.from_pandas(train_data), train_path)
        pq.write_table(pa.Table.from_pandas(test_data), test_path)

        # Write mock parameters to params.yaml
        mock_params = {
            "preprocessing": {
                "training_split": 0.7,
                "validation_split": 0.15,
                "random_state": 42,
            }
        }
        with open(params_path, "w") as param_file:
            yaml.safe_dump(mock_params, param_file)

    yield (
        train_path,
        test_path,
        processed_train_path,
        processed_validation_path,
        processed_test_path,
        params_path,
    )

    # Clean up temporary files
    os.remove(train_path)
    os.remove(test_path)
    os.remove(processed_train_path)
    os.remove(processed_validation_path)
    os.remove(processed_test_path)
    os.remove(params_path)


def test_data_preprocessing(setup_environment):
    """Test the data preprocessing functionality."""
    (
        train_path,
        test_path,
        processed_train_path,
        processed_validation_path,
        processed_test_path,
        params_path,
    ) = setup_environment

    # Patch to replace the location of params.yaml in the main function
    with patch(
        "mdsist.preprocessing.open",
        lambda f, *args, **kwargs: (
            open(params_path, *args, **kwargs) if f == "params.yaml" else open(f, *args, **kwargs)
        ),
    ):
        from mdsist.preprocessing import main

        main(
            Path(train_path),
            Path(test_path),
            Path(processed_train_path),
            Path(processed_validation_path),
            Path(processed_test_path),
        )

    # Check that the processed files are created
    assert os.path.exists(processed_train_path), "Processed train file was not created."
    assert os.path.exists(processed_validation_path), "Processed validation file was not created."
    assert os.path.exists(processed_test_path), "Processed test file was not created."

    # Load processed data to verify splits
    train_df = pq.read_table(processed_train_path).to_pandas()
    validation_df = pq.read_table(processed_validation_path).to_pandas()
    test_df = pq.read_table(processed_test_path).to_pandas()

    # Verify the shapes of the splits
    assert (
        len(train_df) + len(validation_df) + len(test_df) == 15
    ), "Total number of records is incorrect."
    assert len(train_df) == 10, "Training split size is incorrect."
    assert len(validation_df) == 2, "Validation split size is incorrect."
    assert len(test_df) == 3, "Test split size is incorrect."


def test_main_with_invalid_data(setup_environment):
    (
        train_path,
        test_path,
        processed_train_path,
        processed_validation_path,
        processed_test_path,
        params_path,
    ) = setup_environment

    # Patch to replace the location of params.yaml in the main function
    with patch(
        "mdsist.preprocessing.open",
        lambda f, *args, **kwargs: (
            open(params_path, *args, **kwargs) if f == "params.yaml" else open(f, *args, **kwargs)
        ),
    ):
        # Provide invalid input to test error handling
        invalid_train_path = "invalid/path/to/train"
        from mdsist.preprocessing import main

        with pytest.raises(FileNotFoundError):
            main(
                invalid_train_path,
                test_path,
                processed_train_path,
                processed_validation_path,
                processed_test_path,
            )
