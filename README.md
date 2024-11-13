# MDSIST

The aim of the project is to implement a CNN  model for image classification on the MNIST dataset.  The MNIST dataset is widely used in academic and research settings, providing a well-understood and manageable challenge that allows our team to focus on applying MLOps principles rather than data collection or preprocessing challenges.

## Project Setup

### Prerequisites
* Python 3.9
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

## How to run the tests

1. Locate yourself in the root directory

2. Run `coverage run -m pytest`. This will execute all the test files of the project

3. Execute `coverage report` or `coverage html` in order to visually see the results. The latest generates a folder called `htmlcov`. From this folder, `index.html` is the file that contains the results. 
    - In order to see its content, either:
        - Run `open htmlcov/index.html`
        - Manually right click on the file, select `Reveal in File Explorer` and double click the file from the File Explorer
        - If you are using Linux, run `xdg-open htmlcov/index.html`

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
│   ├── 4.0-icc-test
│   ├── 5.0-aag-ji-app
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
│   │
│   ├── __init__.py             <- Makes mdsist a Python module
│   │
│   ├── app.py             
│   │
│   ├── architectures.py        <- Defines a CNN model for image classification in PyTorch
│   │
│   ├── config.py               <- Store useful variables and configuration
│   │
│   ├── dataset.py              <- Scripts to download or generate data
│   │
│   ├── predictor.py             
│   │
│   ├── preprocessing.py        <- Splits raw train and test data into processed train, validation, and test datasets.
│   │
│   ├── trainer.py              <- Trains and validates a model while logging metrics using MLflow.
│   │
│   ├── util.py                 <- Provides utility functions
│   │
│   ├── modeling
│   │   │             
│   │   ├── __init__.py 
│   │   ├── test.py               
│   │   └── train.py     
│   │
│   ├── features
│   │   │                
│       ├── deepchecks_validation.py
│
├── reports  
│   │
│   ├── deepchecks_validation.html
│
├── tests  
│   │
│   ├── mdsist
│   │   │             
│   │   ├── test_api.py
│   │   ├── test_architectures.py
│   │   ├── test_config.py
│   │   ├── test_preprocessing.py
│   │   ├── test_trainer.py
│   │   ├── test_util.py
│ 
├── docker-compose.yml
│
├── Dockerfile
│
├── params.yaml
│
├── dvc.yaml
│
├── pynblint_pre_commit.py
```

--------

