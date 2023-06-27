#Test model

import pytest
from starter.ml.data_process import load_data, process_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib