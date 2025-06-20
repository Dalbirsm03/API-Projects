import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import uvicorn

with open("classifier.pkl", "rb") as pickle_in:
    classifier = pickle.load(pickle_in)

app = FastAPI()

class Input(BaseModel):
    variance: float
    skewness: float
    curtosis: float
    entropy: float

@app.post("/predict")
def predict(data: Input):
    features = np.array([[
        data.variance,
        data.skewness,
        data.curtosis,
        data.entropy
    ]])
    
    prediction = classifier.predict(features)[0]
    if prediction == 1:
        result = "Fake Note"
    else:
        result = "Genuine Note"
    return {"Authentication": result} 