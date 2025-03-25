import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

def preprocess_predict_data(fileName, num_house_types=5):
    """
    Preprocess prediction data from a given CSV file (which does not include the raw price).
    The function:
      - Applies the same preprocessing (binary conversion, one-hot encoding, numeric conversion).
      - Computes a house score using the predefined FEATURE_WEIGHTS.
      - Loads saved house score bins (from training) and assigns a price category based on the house score.
      - Standardizes the remaining features using the saved scaler.
      - Saves the processed features (and the computed price categories) to a pickle file.
    """
    # Load new prediction data
    df = pd.read_csv(fileName)
    
    # Load environment variables
    load_dotenv()
    
    # Define the same FEATURE_WEIGHTS used in training
    FEATURE_WEIGHTS = {
        "area": float(os.getenv("AREA_CONTRIBUTION", 17)) / 1000,
        "prefarea": int(os.getenv("PREFAREA_CONTRIBUTION", 10)),
        "bedrooms": int(os.getenv("BEDROOMS_CONTRIBUTION", 5)),
        "bathrooms": int(os.getenv("BATHROOMS_CONTRIBUTION", 5)),
        "parking": int(os.getenv("PARKING_CONTRIBUTION", 3)),
        "stories": int(os.getenv("STORIES_CONTRIBUTION", 3)),
        "airconditioning": int(os.getenv("AIRCONDITIONING_CONTRIBUTION", 3)),
        "basement": int(os.getenv("BASEMENT_CONTRIBUTION", 2)),
        "guestroom": int(os.getenv("GUESTROOM_CONTRIBUTION", 2)),
        "mainroad": int(os.getenv("MAINROAD_CONTRIBUTION", -2)),
        "hotwaterheating": int(os.getenv("HOTWATERHEATING_CONTRIBUTION", 1)),
    }
    
    # Convert binary categorical variables ("yes"/"no") to 1/0
    binary_cols = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
    df[binary_cols] = df[binary_cols].applymap(lambda x: 1 if str(x).strip().lower() == "yes" else 0)
    
    if "furnishingstatus" in df.columns:
        df = pd.get_dummies(df, columns=["furnishingstatus"], drop_first=True)
    
    # Convert specified columns to numeric (price is not available in prediction data)
    numeric_cols = ["area", "bedrooms", "bathrooms", "stories", "parking"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Drop rows with missing values
    df = df.dropna()
    
    # Compute house_score using the same weighted sum as in training
    df["house_score"] = sum(
        df[col].astype(float) * FEATURE_WEIGHTS[col]
        for col in FEATURE_WEIGHTS if col in df.columns
    )
    
    # Load the saved house score bins (quantile thresholds computed during training)
    # This ensures consistency with how price categories were assigned during training.
    bins_file = "house_score_bins.pkl"
    if os.path.exists(bins_file):
        house_score_bins = joblib.load(bins_file)
    else:
        # Fallback: compute quantiles from current data (less ideal)
        quantileSet = [i / 5 for i in range(1, 5)]  # assume 5 classes
        house_score_bins = df["house_score"].quantile(quantileSet).values
        joblib.dump(house_score_bins, bins_file)
    
    # Assign price_category based on the house_score and loaded bins
    df["price_category"] = np.digitize(df["house_score"], bins=house_score_bins)
    
    # Prepare features for prediction.
    # Training dropped "price", "price_category", and "house_score" so do the same here.
    drop_cols = ["price_category"]
    X = df.drop(columns=drop_cols, errors="ignore")

    # Load the saved feature column names and reindex the DataFrame accordingly
    feature_columns = joblib.load("feature_columns.pkl")
    X = X.reindex(columns=feature_columns)
    
    # Standardize features using the saved scaler (fitted during training)
    scaler_path = "scaler.pkl"
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X)
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, scaler_path)
    
    # Save the processed features and the computed price categories to disk
    output_file = "preprocessed_predict_data.pkl"
    joblib.dump((X_scaled, df["price_category"].values), output_file)
    
    return output_file

if __name__ == "__main__":
    # For testing: ensure you have a CSV (e.g., "new_data.csv") with the same features (without price).
    output = preprocess_predict_data("new_data.csv", num_house_types=5)
    print("Preprocessed prediction data saved to:", output)