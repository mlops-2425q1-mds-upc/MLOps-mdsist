import pytest
import torch
from torch.autograd import Variable
from mdsist.architectures import CNN

@pytest.fixture
def cnn_model():
    """Fixture to instantiate the CNN model."""
    return CNN()

def test_model_structure(cnn_model):
    """Test that the model layers are correctly initialized."""
    assert isinstance(cnn_model.conv1, torch.nn.Conv2d)
    assert isinstance(cnn_model.conv2, torch.nn.Conv2d)
    assert isinstance(cnn_model.fc1, torch.nn.Linear)
    assert isinstance(cnn_model.fc2, torch.nn.Linear)
    assert isinstance(cnn_model.pool, torch.nn.MaxPool2d)

def test_forward_pass(cnn_model):
    """Test the forward pass of the CNN model with a dummy input."""
    batch_size = 8
    input_tensor = torch.randn(batch_size, 1, 28, 28)  # Grayscale images of 28x28 pixels
    output = cnn_model(input_tensor)
    
    # Check output shape: should be (batch_size, 10) as we expect 10 classes
    assert output.shape == (batch_size, 10)

def test_gradient_flow(cnn_model):
    """Test that gradients flow through the model (i.e., backpropagation works)."""
    batch_size = 8
    input_tensor = Variable(torch.randn(batch_size, 1, 28, 28), requires_grad=True)
    output = cnn_model(input_tensor)
    
    # Assume dummy target labels for loss calculation
    target = torch.randint(0, 10, (batch_size,))

    # Use cross-entropy loss for this classification model
    loss_function = torch.nn.CrossEntropyLoss()
    loss = loss_function(output, target)

    # Perform backpropagation
    loss.backward()

    # Check if gradients exist for model parameters
    for param in cnn_model.parameters():
        assert param.grad is not None, "No gradient calculated for a parameter"

def test_parameter_count(cnn_model):
    """Test if the number of model parameters is correct."""
    model_parameters = sum(p.numel() for p in cnn_model.parameters())
    
    # The parameter count is 206922 for this specific CNN architecture
    assert model_parameters == 206922, f"Expected 206922 parameters but got {model_parameters}"

    
if __name__ == "__main__":
    pytest.main()