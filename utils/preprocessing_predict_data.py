import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv

def preprocess_predict_data(fileName):
    """
    Preprocess prediction data from a given CSV file (which does not include the raw price).
    The function:
      - Applies the same preprocessing as training (binary conversion, furnishing status mapping, numeric conversion).
      - For each numeric feature in FEATURE_WEIGHTS (that is not binary), performs min–max normalization (0–1) to compute the house score.
      - Computes the house score as the weighted sum of the normalized features (plus furnishing contribution if available).
      - Loads saved house score bins (from training) and assigns a price category based on the house score.
      - Prepares the prediction feature set (keeping house_score) and normalizes it using the saved MinMaxScaler.
      - Saves the processed features and computed price categories to a pickle file.
    """
    # Load new prediction data
    df = pd.read_csv(fileName)
    
    # Load environment variables
    load_dotenv()
    
    # Get class counts from env variables (defaults if not set)
    num_price_classes = int(os.getenv("PRICE_CLASS_COUNT", 5))
    num_house_types = int(os.getenv("TYPE_CLASS_COUNT", 5))
    
    # Define the same FEATURE_WEIGHTS used in training
    FEATURE_WEIGHTS = {
        "area": float(os.getenv("AREA_CONTRIBUTION", 17)),
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
    
    # Define binary columns (should not be normalized for house score calculation)
    binary_cols = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
    df[binary_cols] = df[binary_cols].applymap(lambda x: 1 if str(x).strip().lower() == "yes" else 0)
    
    # Process furnishing status by mapping its value to a numeric contribution
    if "furnishingstatus" in df.columns:
        furnished_contrib = float(os.getenv("FURNISHINGSTATUS_FURNISHED_CONTRIBUTION", 4))
        semi_contrib = float(os.getenv("FURNISHINGSTATUS_SEMI_FURNISHED_CONTRIBUTION", 2))
        unfurnished_contrib = float(os.getenv("FURNISHINGSTATUS_UNFURNISHED_CONTRIBUTION", 0))
        
        def map_furnishing(value):
            val = str(value).strip().lower()
            if val == "furnished":
                return furnished_contrib
            elif val in ["semi-furnished", "semifurnished"]:
                return semi_contrib
            else:
                return unfurnished_contrib
        
        df["furnishing_contribution"] = df["furnishingstatus"].apply(map_furnishing)
        df.drop(columns=["furnishingstatus"], inplace=True)
    
    # Convert specified columns to numeric (price is not available in prediction data)
    numeric_cols = ["area", "bedrooms", "bathrooms", "stories", "parking"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Drop rows with missing values
    df = df.dropna()

    # --- Compute Normalized Features for House Score Calculation ---
    # Only normalize features in FEATURE_WEIGHTS that are not in binary_cols.
    for col in FEATURE_WEIGHTS:
        if col in df.columns and col not in binary_cols:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max - col_min != 0:
                df[f"{col}_norm"] = (df[col] - col_min) / (col_max - col_min)
            else:
                df[f"{col}_norm"] = 0.5

    # --- Compute House Score ---
    # Multiply each normalized value by its contribution weight and sum them.
    house_score = sum(
        df[f"{col}_norm"] * FEATURE_WEIGHTS[col]
        for col in FEATURE_WEIGHTS if f"{col}_norm" in df.columns
    )
    df["house_score"] = house_score

    # Add furnishing contribution if exists
    if "furnishing_contribution" in df.columns:
        df["house_score"] += df["furnishing_contribution"]

    # --- Generate Targets ---
    # Generate house type by binning house_score
    house_bins = np.linspace(df["house_score"].min(), df["house_score"].max(), num=num_house_types + 1)
    df["house_type"] = np.digitize(df["house_score"], bins=house_bins[:-1], right=True) - 1

    # Derive price category from house_score using quantile thresholds
    quantileSet = [i / num_price_classes for i in range(1, num_price_classes)]
    # Load saved bins if available; otherwise, compute and save them.
    bins_file = "house_score_bins.pkl"
    if os.path.exists(bins_file):
        house_score_bins = joblib.load(bins_file)
    else:
        house_score_bins = df["house_score"].quantile(quantileSet).values
        joblib.dump(house_score_bins, bins_file)
    df["price_category"] = np.digitize(df["house_score"], bins=house_score_bins)

    print("New data house_score min:", df["house_score"].min())
    print("New data house_score max:", df["house_score"].max())
    print("Saved bins:", house_score_bins)

    # --- Prepare Prediction Features ---
    # In training, we dropped "price_category" and "furnishing_contribution", but kept house_score.
    drop_cols = ["price_category"]
    if "furnishing_contribution" in df.columns:
        drop_cols.append("furnishing_contribution")
    # Also drop the normalized columns used for house_score calculation.
    norm_cols = [f"{col}_norm" for col in FEATURE_WEIGHTS if f"{col}_norm" in df.columns]
    drop_cols.extend(norm_cols)
    X = df.drop(columns=drop_cols, errors="ignore")
    # At this point, X should include house_score and the other original numeric columns.

    # Reindex X to match the training features (assuming you saved feature_columns during training)
    feature_columns = joblib.load("feature_columns.pkl")
    X = X.reindex(columns=feature_columns)

    # Normalize the features using the saved MinMaxScaler
    scaler_path = "scaler.pkl"
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X)
    else:
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, scaler_path)

    # Save the processed features and computed price categories to disk
    output_file = "preprocessed_predict_data.pkl"
    joblib.dump((X_scaled, df["price_category"].values), output_file)
    
    return output_file

if __name__ == "__main__":
    # For testing: ensure you have a CSV (e.g., "new_data.csv") with the same features (without price).
    output = preprocess_predict_data("new_data.csv")
    print("Preprocessed prediction data saved to:", output)