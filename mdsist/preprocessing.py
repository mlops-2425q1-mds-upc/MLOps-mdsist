"""
Preprocessing of the data obtained from HuggingFace
"""

import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import typer
import yaml

app = typer.Typer()

"""
Data Preprocessing

This script reads raw training and testing data, merges them, and splits them into
training, validation, and test datasets. The processed datasets are then saved 
as parquet files.
"""


@app.command()
def main(
    raw_train_data: Path,
    raw_test_data: Path,
    processed_train_split: Path,
    processed_validation_split: Path,
    processed_test_split: Path,
):
    """Preprocess raw data and split into training, validation, and test sets.

    Args:
        raw_train_data (Path): Path to the raw training data file.
        raw_test_data (Path): Path to the raw testing data file.
        processed_train_split (Path): Path to save the processed training data.
        processed_validation_split (Path): Path to save the processed validation data.
        processed_test_split (Path): Path to save the processed test data.
    """
    # Load preprocessing parameters from YAML file
    with open("params.yaml", encoding="UTF-8") as param_file:
        params = yaml.safe_load(param_file)["preprocessing"]

    # Read raw test and train files, and merge them together
    files = [raw_train_data, raw_test_data]
    tables = [pq.read_table(file) for file in files]
    combined_table = pa.concat_tables(tables)

    # Convert the combined table to a pandas DataFrame
    df = combined_table.to_pandas()

    # Shuffle and split the DataFrame into train, validation, and test sets
    train, validation, *test = np.split(
        df.sample(frac=1, random_state=params["random_state"]),
        [
            int(params["training_split"] * len(df)),
            int((params["training_split"] + params["validation_split"]) * len(df)),
        ],
    )

    # Reset the index of the splits
    train = train.reset_index(drop=True)
    validation = validation.reset_index(drop=True)
    test = test[0].reset_index(drop=True)

    # Ensure output folders exist
    os.makedirs(processed_train_split.parent, exist_ok=True)
    os.makedirs(processed_validation_split.parent, exist_ok=True)
    os.makedirs(processed_test_split.parent, exist_ok=True)

    # Save the processed datasets to parquet files
    pq.write_table(pa.Table.from_pandas(train), processed_train_split)
    pq.write_table(pa.Table.from_pandas(validation), processed_validation_split)
    pq.write_table(pa.Table.from_pandas(test), processed_test_split)


if __name__ == "__main__":
    app()
