"""
Testing module for the api
"""

import pytest
from fastapi.testclient import TestClient

from mdsist.app import app
from mdsist.config import PROCESSED_DATA_DIR
from mdsist.dataset import MdsistDataset


def get_img_bytes(ind):
    """Gets the label and bytes of the image indexed by ind in the test dataset"""

    test_dataset = MdsistDataset(PROCESSED_DATA_DIR / "test.parquet")
    image_dict = test_dataset.data.loc[ind, "image"]
    image_array = test_dataset.decode_png_image(image_dict)
    label = test_dataset.data.loc[ind, "label"]
    image = image_array.tobytes()
    return image, label


def get_multiple_img_bytes(inds):
    """Returns the concatenation of the bytes and the
    array of labels of all the images indexed by inds
    (index array)"""

    images = bytes(0)
    labels = []

    for ind in inds:
        image, label = get_img_bytes(ind)
        images += image
        labels.append(label)

    return images, (labels)


@pytest.fixture(scope="module", autouse=True)
def client():
    """Use the TestClient with a `with` statement to trigger the startup and shutdown events."""
    with TestClient(app) as client:
        yield client


def test_root(client):
    """Testing of the root endpoint"""
    response = client.get("/")
    json = response.json()
    assert response.status_code == 200
    assert json["data"]["message"] == "Welcome. See /docs for more information about the api"
    assert json["message"] == "OK"
    assert json["status-code"] == 200


def test_get_info(client):
    """Testing of the info endpoint"""

    desc = (
        "The primary intended use of this model is to classify"
        "images of handwritten digits from the MNIST dataset"
        "into one of ten categories (0-9). It was specifically "
        "designed for image classification tasks without"
        "requiring additional fine-tuning or integration into larger applications. "
        "This model is ideal for educational, research, and benchmarking purposes within "
        "the field of machine learning, "
        "particularly in the area of digit recognition."
    )

    short = (
        "This is a Convolutional Neural Network (CNN) model to classify grayscale images "
        "from the MNIST dataset."
    )

    response = client.get("/info")
    json = response.json()
    assert response.status_code == 200
    assert json["data"] == {
        "Name": "MDSIST-CNN",
        "Short Description": short,
        "Description": desc,
        "Layers": {
            "CNN (CNN)": [
                "Conv2d (conv1): 1-1",
                "Conv2d (conv2): 1-2",
                "Linear (fc1): 1-3",
                "Linear (fc2): 1-4",
                "MaxPool2d (pool): 1-5",
            ]
        },
        "Total parameters": 206922,
        "Trainable params": 206922,
        "Total param bytes": 827688,
    }

    assert json["message"] == "OK"
    assert json["status-code"] == 200


@pytest.mark.parametrize(
    ["sample", "expected"],
    [get_img_bytes(i) for i in [85, 60]],
)
def test_predict_one_number(client, sample, expected):
    """Testing of the predict endpoint with one image"""

    headers = {"Content-Type": "application/octet-stream"}
    response = client.post(
        "/mnist-model-prediction",
        headers=headers,
        content=sample,
        timeout=30,
    )

    json = response.json()
    assert response.status_code == 200
    assert json["message"] == "OK"
    assert json["status-code"] == 200
    assert json["data"]["prediction"][0] == expected


@pytest.mark.parametrize(
    ["samples", "expectations"],
    [(get_multiple_img_bytes([85, 60]))],
)
def test_predict_multiple_numbers(client, samples, expectations):
    """Testing of the predict endpoint with multiple images"""
    headers = {"Content-Type": "application/octet-stream"}
    response = client.post(
        "/mnist-model-prediction",
        headers=headers,
        content=samples,
        timeout=30,
    )

    json = response.json()
    assert response.status_code == 200
    assert json["message"] == "OK"
    assert json["status-code"] == 200
    assert json["data"]["prediction"] == expectations


if __name__ == "__main__":
    pytest.main()
