#POSTS to the API

import requests

url = "https://census-income-predict.onrender.com/predict"
sample = {
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

r = requests.post(url, json=sample)

print(f"Status code: {r.status_code}")
print(f"Body: {r.json()}")

