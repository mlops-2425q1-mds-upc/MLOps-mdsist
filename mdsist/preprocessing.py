import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
import sys
import typer
from pathlib import Path

app = typer.Typer()



@app.command()
def main(
    raw_train_data: Path,
    raw_test_data: Path,
    processed_train_split: Path,
    processed_validation_split: Path,
    processed_test_split: Path,

):

    params = yaml.safe_load(open("params.yaml"))["preprocessing"]

    # Read test and train file, and then merge it together
    files = [raw_train_data, raw_test_data]
    tables = [pq.read_table(file) for file in files]
    combined_table = pa.concat_tables(tables)
    #pq.write_table(combined_table, "../data/raw/raw_data.parquet")


    df = combined_table.to_pandas() # pd.read_parquet(combined_table, engine="pyarrow")

    train, validation, test = \
                np.split(df.sample(frac=1, random_state=params["random_state"]), 
                        [int(params["training_split"]*len(df)), int((params["training_split"]+params["validation_split"])*len(df))])

    train = train.reset_index(drop=True)
    validation = validation.reset_index(drop=True)
    test = test.reset_index(drop=True)

    pq.write_table(pa.Table.from_pandas(train), processed_train_split)
    pq.write_table(pa.Table.from_pandas(validation), processed_validation_split)
    pq.write_table(pa.Table.from_pandas(test), processed_test_split)

if __name__ == "__main__":
    app()