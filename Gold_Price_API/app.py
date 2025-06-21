from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

with open("classifier.pkl", "rb") as pickle_in:
    classifier = pickle.load(pickle_in)

app = FastAPI()

class InputData(BaseModel):
    SPX: float
    USO: float
    SLV: float

@app.post("/predict")
def prediction(data: InputData):
    features = np.array([[
        data.SPX,
        data.USO,
        data.SLV
    ]]).reshape(1, -1)
    prediction = classifier.predict(features)[0]
    return {"Gold Price": float(prediction)}


