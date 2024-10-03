# MDSIST

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Machine Learning project around MNIST dataset

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
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

