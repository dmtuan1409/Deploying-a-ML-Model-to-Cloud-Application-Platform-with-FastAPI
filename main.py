# Put the code for your API here.

from fastapi import FastAPI
import pandas as pd    
import numpy as np
from pydantic import BaseModel
import joblib
from starter.ml.data_process import process_data, cat_features
from starter.ml.model import inference

column = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours-per-week",
        "native-country"]

class InputData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        schema_extra = {
            "example": {
                "age": 35,
                "workclass": "Private",
                "fnlgt": 77516,
                "education": "HS-grad",
                "education_num": 9,
                "marital_status": "Divorced",
                "occupation": "Handlers-cleaners",
                "relationship": "Husband",
                "race": "Black",
                "sex": "Male",
                "capital_gain": 0,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States"
            }
        }

# create app
app = FastAPI()
# load models
model = joblib.load("model/model.pkl")
encoder = joblib.load("model/encoder.pkl")
lb = joblib.load("model/lb.pkl")

#GET on the root giving a welcome message.
@app.get("/")
async def root():
    return {"Hello world!"}

#POST that does model inference.
@app.post("/predict")
async def predict(input_data: InputData):
    input = np.array([[
                        input_data.age,
                        input_data.workclass,
                        input_data.fnlgt,
                        input_data.education,
                        input_data.education_num,
                        input_data.marital_status,
                        input_data.occupation,
                        input_data.relationship,
                        input_data.race,
                        input_data.sex,
                        input_data.capital_gain,
                        input_data.capital_loss,
                        input_data.hours_per_week,
                        input_data.native_country
                    ]])
    data = pd.DataFrame(data=input, columns=column)
    #Process the data
    X, _, _, _ = process_data(
                    data, categorical_features=cat_features, training=False, encoder=encoder, lb=lb
                )
    #Inference 
    y = inference(model=model, X=X)
    output = lb.inverse_transform(y)[0]
    return {"Output model: ",output}
