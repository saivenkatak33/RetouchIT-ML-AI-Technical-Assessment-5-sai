# preprocessing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(data: pd.DataFrame):
    """
    Preprocess the fraud dataset for model prediction.
    """

    # Drop columns not useful for training
    cols_to_drop = ["nameOrig", "nameDest", "isFlaggedFraud"]
    data = data.drop(columns=cols_to_drop, errors="ignore")

    # Encode categorical column 'type'
    if "type" in data.columns:
        le = LabelEncoder()
        data["type"] = le.fit_transform(data["type"])

    # Split features (X) and target (y) if target exists
    if "isFraud" in data.columns:
        X = data.drop(columns=["isFraud"])
        y = data["isFraud"]
        return X, y
    else:
        # If no target column, just return features (for prediction only)
        return data
