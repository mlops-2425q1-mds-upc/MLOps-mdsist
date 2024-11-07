from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn

app = FastAPI()

## Saps on tenim el nostre model entrenat? we can import it with pickel or joblib
model = joblib.dump("our_trained_mnist_model")

class PredictionRequest(BaseModel):
    image: list # The input data should be an array list of 256 pixel

@app.post("/mnist-model-prediction/")
async def predict(data: PredictionRequest):
    image_array = np.array(data.image).reshape(1,-1)
    prediction = model.predict(image_array)
    return{"prediction": int(prediction[0])}

