from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()

try:
    model = joblib.load("xgb_model.pkl")
except Exception as e:
    print("❌ Error loading model:", e)

class InputData(BaseModel):
    Gender: str
    Age: int
    Height: float
    Weight: float
    Duration: float
    Heart_Rate: float
    Body_Temp: float

@app.post("/predict")
def predict(data: InputData):
    try:
        print("📥 Received input:", data)

        gender_val = 1 if data.Gender.lower() == "male" else 0

        features = np.array([
            gender_val,
            data.Age,
            data.Height,
            data.Weight,
            data.Duration,
            data.Heart_Rate,
            data.Body_Temp
        ]).reshape(1, -1)

        print("🧮 Features array:", features)

        prediction = float(model.predict(features)[0])

        print("✅ Prediction:", prediction)

        return {"predicted_calories_burned": round(prediction, 2)}

    except Exception as e:
        print("❌ Prediction error:", e)
        return {"error": "Prediction failed", "details": str(e)}
