"""
Tests for the Trainer module in the mdsist project.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from mdsist.architectures import CNN
from mdsist.dataset import MdsistDataset
from mdsist.trainer import Trainer


@pytest.fixture
def mock_data_loader():
    """
    Fixture to create mock DataLoaders for training and validation using the MNIST dataset.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]  # Normalize dataset
    )

    root_dir = Path(__file__).parent.parent.parent
    path_to_train = os.path.join(root_dir, "data/processed/train.parquet")
    path_to_val = os.path.join(root_dir, "data/processed/validation.parquet")

    train_dataset = MdsistDataset(path_to_train, transform=transform)
    val_dataset = MdsistDataset(path_to_val, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader


@pytest.fixture
def mock_model():
    """
    Fixture to create a CNN model.
    """
    return CNN()


@pytest.fixture
def mock_optimizer(mock_model):
    """
    Fixture to create an optimizer for the model.
    """
    return torch.optim.Adam(mock_model.parameters(), lr=0.001)


@pytest.fixture
def trainer(mock_model, mock_optimizer):
    """
    Fixture to create a Trainer instance.
    """
    trainer = Trainer(model=mock_model, optimizer=mock_optimizer, device="cpu")
    reset_model_weights(trainer.model)  # Ensure weights are reset
    return trainer


def test_data_loader(mock_data_loader):
    """
    Test data loader is correctly defined
    """
    train_loader, val_loader = mock_data_loader

    assert len(train_loader.dataset) > 0, "The train dataset is empty!"
    assert len(val_loader.dataset) > 0, "The validation dataset is empty!"

    assert (
        train_loader.batch_size == 64
    ), f"Expected batch size of 64, but got {train_loader.batch_size}"
    assert (
        val_loader.batch_size == 64
    ), f"Expected batch size of 64, but got {val_loader.batch_size}"


def reset_model_weights(model):
    """
    Reset the model weights by reinitializing each layer.
    """
    for layer in model.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()


def test_trainer_initialization(mock_model, mock_optimizer):
    """
    Test the initialization of the Trainer class.
    """
    trainer = Trainer(model=mock_model, optimizer=mock_optimizer, device="cpu")
    assert trainer.model == mock_model
    assert trainer.optimizer == mock_optimizer
    assert trainer.device == "cpu"


def test_minimum_functionality(trainer, mock_data_loader):
    """
    Test minimum functionality to ensure the model can complete a training epoch.
    """
    reset_model_weights(trainer.model)

    train_loader, _ = mock_data_loader
    train_stats = trainer.train_epoch(train_loader)

    assert isinstance(train_stats.loss, float), "Loss is not a float."
    assert train_stats.loss >= 0, "Loss should not be negative."
    assert train_stats.accuracy >= 0, "Accuracy should not be negative."


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_training_on_different_devices(device, trainer, mock_data_loader):
    """
    Test that the training works on CPU and CUDA, if available.
    """
    reset_model_weights(trainer.model)

    train_loader, _ = mock_data_loader
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    trainer.device = device
    trainer.model.to(device)

    train_stats = trainer.train_epoch(train_loader)
    assert train_stats.accuracy >= 0.0, "Training failed, accuracy is NaN."


def test_model_invariance(trainer):
    """
    Test model invariance by ensuring outputs are consistent for the same input.
    """
    input_tensor = torch.randn(2, 1, 28, 28)

    outputs_1 = trainer.model(input_tensor)
    outputs_2 = trainer.model(input_tensor)

    assert torch.equal(outputs_1, outputs_2), "Model outputs are not invariant for the same input."


def test_full_training_cycle(trainer, mock_data_loader):
    """
    Integration test for the entire training and validation cycle.
    """
    reset_model_weights(trainer.model)

    torch.manual_seed(42)

    train_loader, val_loader = mock_data_loader
    initial_val_stats = trainer.validate(val_loader)

    for _ in range(5):
        train_stats = trainer.train_epoch(train_loader)
        assert train_stats.accuracy >= 0, "Accuracy should not be negative."
        trainer.validate(val_loader)

    val_stats = trainer.validate(val_loader)
    assert (
        val_stats.accuracy > initial_val_stats.accuracy
    ), "Validation accuracy did not improve after training."
    assert (
        val_stats.loss < initial_val_stats.loss
    ), "Validation loss did not decrease after training."
    assert val_stats.accuracy > 0.9, "Minimum accuracy considered in our Project."


def test_single_batch_training(trainer):
    """
    Test training with a DataLoader that contains only a single batch.
    """
    reset_model_weights(trainer.model)
    input_tensor = torch.randn(2, 1, 28, 28)
    labels = torch.tensor([0, 1])
    single_batch_dataset = TensorDataset(input_tensor, labels)
    single_batch_loader = DataLoader(single_batch_dataset, batch_size=2)

    train_stats = trainer.train_epoch(single_batch_loader)

    assert isinstance(train_stats.loss, float), "Loss should be a float."
    assert train_stats.accuracy >= 0, "Accuracy should not be negative."


def test_logging_metrics(trainer, mock_data_loader):
    """
    Test that MLflow logs the metrics correctly.
    """
    reset_model_weights(trainer.model)
    train_loader, val_loader = mock_data_loader

    with (
        patch("mlflow.start_run") as mock_start_run,
        patch("mlflow.end_run") as _,
        patch("mlflow.log_metric") as mock_log_metric,
    ):

        trainer.train(train_loader, val_loader, epochs=1)

        # Verificar que mlflow.start_run no fue llamado
        mock_start_run.assert_not_called()

        # Verificar que se están registrando las métricas
        assert mock_log_metric.call_count > 0, "MLflow did not log any metrics."
        assert any(
            "train_accuracy" in call[0][0] for call in mock_log_metric.call_args_list
        ), "train_accuracy not logged."


if __name__ == "__main__":
    pytest.main()
