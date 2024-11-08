import joblib
import numpy as np
import uvicorn
import mlflow
import mdsist.util as util

from fastapi import FastAPI
from pydantic import BaseModel

from dotenv import load_dotenv

# Load envionment variables
load_dotenv()

app = FastAPI()

## Import model from MLFlow, with below URI
MODEL_URI = 'runs:/10cb51b288134c48835a8c0b9fe66eca/model_20240930190709'
#'models:/CNNv1-production/'

device = util.get_available_device()

model = mlflow.pytorch.load_model(MODEL_URI, map_location=device)


class PredictionRequest(BaseModel):
    image: list # The input data should be an array list of 256 pixel

@app.post("/mnist-model-prediction/")
async def predict(data: PredictionRequest):
    # pass the image as byte, then from buffer.
    
    np.frombuffer()

    # reshape uint8 to below structure
    shape (N, 1, H, W) # N = degree of freedom, -1.  H = 28, W = 28
    image_array = np.array(data.image).reshape(1,-1)
    prediction = model.predict(image_array)
    return{"prediction": int(prediction[0])}

