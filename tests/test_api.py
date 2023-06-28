from fastapi.testclient import TestClient

# Import our app from main.py.
from main import app

# Instantiate the testing client with our app.
client = TestClient(app)

def test_get_root():
    """
        Test get root
    """
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()[0] == "Hello world!"

def test_post_inference():
    """
        test_model inference
    """
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
    r = client.post("/predict", json=sample)
    assert r.status_code == 200
    assert r.json() == "<=50K" 
    
def test_post_inference_false_query():
    """
        test_model inference with false query
    """
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
                "sex": "Male"
            }
    r = client.post("/predict", json=sample)
    assert 'capital_gain' not in r.json()
    assert 'capital_loss' not in r.json()
    assert 'hours_per_week' not in r.json()
    assert 'native_country' not in r.json()