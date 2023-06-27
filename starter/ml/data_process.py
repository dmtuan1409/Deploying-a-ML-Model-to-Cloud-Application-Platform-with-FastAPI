#Script for load data, process data and caculate performance of the model on slices of the data

import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
import pandas as pd 

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

def load_data(path):
    """load data

    Args:
        path (str): data directory

    Returns:
        data (dataframe): 
    """
    #Read data
    data = pd.read_csv(path, skipinitialspace=True)
    #Drop NaN value
    data = data.replace("?", None).dropna()
    return data




def process_data(
    X, categorical_features=[], label=None, training=True, encoder=None, lb=None
):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb


def slice_performance(test_data, model, encoder, lb, compute_model_metrics, cat_features=cat_features):
    """Caculate performance of the model on slices of the data

    Args:
        test_data (dataframe): test data
        model (sklearn model): model
        encoder (sklearn.preprocessing._encoders.OneHotEncoder): Trained sklearn OneHotEncoder
        lb (sklearn.preprocessing._label.LabelBinarizer): Trained LabelBinarizer
        compute_model_metrics (function caculate precision, recall, fbeta): Validates the trained machine learning model using precision, recall, and F1.
        cat_features (list, optional): category feature. Defaults to cat_features.
    Return:
        file: slice_performance.txt
    """
    with open('slice_performance.txt', 'w') as f:
        #Loop all categorical features
        for cat in cat_features:
            #Loop each value in each categorical feature
            for value in test_data[cat].unique():
                #Get slices of the data
                slice_data = test_data[test_data[cat]==value].reset_index(drop=True)
                # Proces the slice_data with the process_data function.
                X_test, y_test, _, _ = process_data(
                    slice_data, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
                )
                #Predict X_test
                y_pred = model.predict(X_test)
                #Evaluation
                precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
                row = "{cat}:{value}, Precision:{precision}, Recall:{recall}, Fbeta:{fbeta}\n".format(cat=cat, value=value, precision=round(precision,2), recall=round(recall,2), fbeta=round(fbeta,2))
                f.write(row)