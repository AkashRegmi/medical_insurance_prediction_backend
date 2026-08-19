from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("insurance_model.pkl", "rb") as file:
    model = pickle.load(file)
    
class InsuranceInput(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str
    
@app.get("/")
def home():
    return {
        "message": "Insurance Prediction API is running"
    }
    
@app.post("/predict")
def predict_insurance(data: InsuranceInput):

    # Encoding dictionaries
    sex_mapping = {
        "male": 0,
        "female": 1
    }

    smoker_mapping = {
        "yes": 0,
        "no": 1
    }
     
    region_mapping = {
        "southwest": 0,
        "southeast": 1,
        "northwest": 2,
        "northeast": 3
    }

    # Convert user input into model values
    input_data = np.array([
        [
            data.age,
            sex_mapping[data.sex.lower()],
            data.bmi,
            data.children,
            smoker_mapping[data.smoker.lower()],
            region_mapping[data.region.lower()]
        ]
    ])
    prediction = model.predict(input_data)
    
    return {
        "predicted_insurance_charge": round(float(prediction[0]), 2)
    }