#Test model
import sys
sys.path.append('../')
import pytest
from starter.ml.data_process import load_data, process_data, cat_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
from starter.ml.model import inference


@pytest.fixture(scope="module")
def data():
    # code to load in the data.
    datapath = "./data/census.csv"
    return load_data(datapath)

def test_load_data(data):
    """
    Check data
    """
    assert data.shape[0] > 0
    assert data.shape[1] > 0

def test_process_data(data):
    """test data train test split

    Args:
        data (dataframe): read data census.csv
    """
    train, test = train_test_split(data, test_size=0.3, random_state=0)
    # Process data
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_train) + len(X_test) == len(data)
    
def test_inference(data):
    """Test inference function

    Args:
        data (dataframe): read data census.csv
    """
    model = joblib.load(r'./model/model.pkl')
    train, test = train_test_split(data, test_size=0.3, random_state=0)
    # Process data
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    y_preds = inference(model, X_test)
    assert len(y_preds) == len(test)