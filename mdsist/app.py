"""
api module
"""

import io
from http import HTTPStatus

import mlflow
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torch import Tensor
from torchinfo import ModelStatistics, summary
from torchinfo.layer_info import LayerInfo
from torchvision import transforms

from mdsist import util
from mdsist.predictor import Predictor

# Load envionment variables
load_dotenv()

app = FastAPI()

## Import model from MLFlow, with below URI
MODEL_URI = "runs:/10cb51b288134c48835a8c0b9fe66eca/model_20240930190709"
#'models:/CNNv1-production/'

device = util.get_available_device()

model = mlflow.pytorch.load_model(MODEL_URI, map_location=device)

pred = Predictor(model)


@app.get("/")
async def root():
    """
    Gives information where to find documentation
    """
    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"message": "Welcome. See /docs for more information about the api"},
    }


@app.get("/info")
async def model_info():
    """
    return info about the model
    """
    info: ModelStatistics = summary(model, verbose=0)
    layers: list[LayerInfo] = info.summary_list

    desc = (
        "The primary intended use of this model is to classify "
        "images of handwritten digits from the MNIST dataset "
        "into one of ten categories (0-9). It was specifically "
        "designed for image classification tasks without "
        "requiring additional fine-tuning or integration into larger applications. "
        "This model is ideal for educational, research, and benchmarking purposes within "
        "the field of machine learning, "
        "particularly in the area of digit recognition."
    )

    short = (
        "This is a Convolutional Neural Network (CNN) model to classify grayscale images "
        "from the MNIST dataset."
    )

    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {
            "Name": "MDSIST-CNN",
            "Short Description": short,
            "Description": desc,
            "Layers": {
                layers[0].get_layer_name(True, True): [
                    layer.get_layer_name(True, True) for layer in layers[1:]
                ]
            },
            "Total parameters": info.total_params,
            "Trainable params": info.trainable_params,
            "Total param bytes": info.total_param_bytes,
        },
    }


@app.post("/mnist-model-prediction")
async def predict(files: list[UploadFile] = File(...)):
    """
    predicts from png image
    """
    print(format)

    def png_to_bytearray(pngimage):
        # Create a BytesIO object from the binary data
        image_stream = io.BytesIO(pngimage)

        # Open the image using Pillow
        with Image.open(image_stream) as img:
            img = img.convert("L")
            # Convert the image to a NumPy array
            return np.array(img)

    image_array = [png_to_bytearray(await file.read()) for file in files]

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    imgs_tensor = []
    for img in image_array:
        img_reshape = img.reshape((1, 28, 28))
        imgs_tensor.append(transform(img_reshape))

    imgs_tensor_reshape = Tensor(np.array(imgs_tensor)).reshape((-1, 1, 28, 28))
    prediction = pred.predict(imgs_tensor_reshape)

    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"prediction": [int(i) for i in prediction]},
    }
