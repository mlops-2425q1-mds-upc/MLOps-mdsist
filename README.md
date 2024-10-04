# MDSIST

The aim of the project is to implement a CNN  model for image classification on the MNIST dataset.  The MNIST dataset is widely used in academic and research settings, providing a well-understood and manageable challenge that allows our team to focus on applying MLOps principles rather than data collection or preprocessing challenges.

## Project Setup

### Prerequisites
* Python 3.11 or higher
* Poetry Python package

### Steps

1. Install dependences:
```shell
poetry install
```
2. Edit project parameters present on `params.yaml` as you wish.

3. Use `dist.env` as a template to create a `.env` file with the corresponding information

4. Execute DVC pipeline:

```shell
dvc repro
```

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── processed      <- The final, canonical data sets for modeling.
│   │   ├── test.parquet
│   │   ├── train.parquet
│   │   ├── validation.parquet
│   │
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks.
│   │
│   ├── 1.0-ji-raw-data-preprocessing
│   ├── 1.0-ji-raw-data-visualization
│   ├── 2.0-ji-raw-data-split
│   ├── 3.0-icc-train
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         mdsist and configuration for tools like black
│
│
└── cards    <- Contains the dataset and model cards
│   │
│   ├── mnist_data_card
│   ├── model_card
│
└── mdsist   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mdsist a Python module
    │
    ├── architectures.py        <- Defines a CNN model for image classification in PyTorch
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── preprocessing.py        <- Splits raw train and test data into processed train, validation, and test datasets.
    │
    ├── trainer.py              <- Trains and validates a model while logging metrics using MLflow.
    │
    ├── util.py                 <- Provides utility functions
    │
    ├── modeling                
        ├── __init__.py 
        ├── predict.py          <- Code to run model inference with trained models          
        └── train.py            <- Code to train models
```

--------

