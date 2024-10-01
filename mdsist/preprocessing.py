import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
import sys

def main():

    params = yaml.safe_load(open("params.yaml"))["preprocessing"]

    if len(sys.argv) != 6:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython prepare.py raw-train-data.parquet raw-test-data.parquet processed-train-split processed-validation-split processed-test-split\n")
        sys.exit(1)

    # Read test and train file, and then merge it together
    files = [sys.argv[1], sys.argv[2]]
    tables = [pq.read_table(file) for file in files]
    combined_table = pa.concat_tables(tables)
    #pq.write_table(combined_table, "../data/raw/raw_data.parquet")


    df = combined_table# pd.read_parquet(combined_table, engine="pyarrow")

    train, validation, test = \
                np.split(df.sample(frac=1, random_state=params["random_state"]), 
                        [int(params["training_split"]*len(df)), int((params["training_split"]+params["validation_split"])*len(df))])

    train = train.reset_index(drop=True)
    validation = validation.reset_index(drop=True)
    test = test.reset_index(drop=True)

    pq.write_table(pa.Table.from_pandas(train), sys.argv[3])
    pq.write_table(pa.Table.from_pandas(validation), sys.argv[4])
    pq.write_table(pa.Table.from_pandas(test), sys.argv[5])

if __name__ == "__main__":
    main()