"""
api module
"""

import mlflow
import numpy as np
import torch
from dotenv import load_dotenv
from fastapi import Body, FastAPI

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


@app.get("/test")
async def test():
    """
    test function
    """
    return "It works"


@app.post("/mnist-model-prediction")
async def predict(data: bytes = Body(...)):
    """
    predicts from image
    """

    # pass the image as byte, then from buffer.

    data_array = bytearray(data)

    image_array = torch.tensor(data_array, dtype=torch.uint8)

    # reshape uint8 to below structure
    # shape (N, 1, H, W) # N = degree of freedom, -1.  H = 28, W = 28
    images = image_array.reshape((-1, 1, 28, 28))
    print(np.shape(images))

    prediction = pred.predict(images)
    print(prediction)
    return {"prediction": [int(i) for i in prediction]}
