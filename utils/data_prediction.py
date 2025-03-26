import os
import joblib
import pandas as pd
import numpy as np

def predict_model(model_name, prediction_type, preprocessed_file):
    """
    Loads preprocessed prediction data (normalized using MinMaxScaler),
    reindexes it using saved feature columns, loads the specified model,
    fills any missing values with 0, and returns predictions.
    
    Parameters:
      model_name: str - one of "random_forest", "logistic_regression", "gradient_boosting"
      prediction_type: str - target type (e.g., "price" or "type")
      preprocessed_file: str - path to the pickle file containing the preprocessed prediction data.
          Expected to be a tuple where the first element is the normalized feature matrix.
    
    Returns:
      List of predictions, or None if an error occurs.
    """
    # Load the preprocessed data (assumed to be a tuple: (X_scaled, additional_info))
    data = joblib.load(preprocessed_file)
    
    # Extract the feature matrix (X_scaled). If data is a tuple, assume the first element is X_scaled.
    if isinstance(data, tuple):
        X_data = data[0]
    else:
        X_data = data

    # If X_data is a DataFrame, reindex it using the saved feature columns.
    if isinstance(X_data, pd.DataFrame):
        feature_columns_file = "feature_columns.pkl"
        if os.path.exists(feature_columns_file):
            feature_columns = joblib.load(feature_columns_file)
            X_data = X_data.reindex(columns=feature_columns)
            # Fill missing columns (NaN values) with 0
            X_data = X_data.fillna(0)
        else:
            print("Feature columns file not found.")
    
    # If X_data is a numpy array, ensure there are no NaNs.
    if isinstance(X_data, np.ndarray):
        X_data = np.nan_to_num(X_data, nan=0.0)
    
    # Determine the model file path based on the model name and prediction type.
    MODEL_DIR = os.getenv("MODEL_DIR")
    if model_name == "random_forest":
        model_file = os.path.join(MODEL_DIR, f"random_forest_{prediction_type}.pkl")
    elif model_name == "logistic_regression":
        model_file = os.path.join(MODEL_DIR, f"logistic_regression_{prediction_type}.pkl")
    elif model_name == "gradient_boosting":
        model_file = os.path.join(MODEL_DIR, f"gradient_boosting_{prediction_type}.pkl")
    else:
        print("Invalid model name provided.")
        return None

    try:
        model = joblib.load(model_file)
        predictions = model.predict(X_data).tolist()
        return predictions
    except Exception as e:
        print("Prediction error:", e)
        return None

# Example usage:
# preds = predict_model("random_forest", "price", "preprocessed_predict_data.pkl")
# print("Predictions:", preds)