# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from data_process import load_data, process_data, slice_performance
from model import train_model, compute_model_metrics, inference
import joblib

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Add code to load in the data.
data = load_data(path = r'../../data/census.csv')
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)
# Process data
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)
# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    train, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)
# Train and save a model.
print("Starting training")
model = train_model(X_train, y_train)
print("Inference")
y_pred = inference(model, X_test)
print("Evaluation")
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
print("Precision: ",round(precision,2)," Recall: ",round(recall,2)," FBeta: ",round(fbeta,2))
# test on slices of the data
print("Caculate performance of the model on slices of the data")
slice_performance(test, model, encoder, lb, compute_model_metrics, cat_features=cat_features)

# Save model
print("Saving model")
joblib.dump(model, '../../model/model.pkl')
joblib.dump(encoder, '../../model/encoder.pkl')
joblib.dump(lb, '../../model/lb.pkl')
