import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv

def preprocess_data(num_price_classes=5, num_house_types=5, fileName="Housing_Augmented.csv"):
    # Load dataset
    df = pd.read_csv(fileName)

    # Load environment variables
    load_dotenv()

    # Define contribution weights (used to compute the house score) for numeric features
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

    # Define binary columns (they are converted to 0/1 and should not be normalized in this step)
    binary_cols = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
    df[binary_cols] = df[binary_cols].applymap(lambda x: 1 if str(x).strip().lower() == "yes" else 0)

    # Furnishing status processing:
    # Instead of one-hot encoding, map the furnishing status to a numeric contribution.
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

    # Convert specified columns to numeric
    numeric_cols = ["price", "area", "bedrooms", "bathrooms", "stories", "parking"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing values
    df = df.dropna()

    # --- Compute Normalized Features for House Score Calculation ---
    # Only normalize those features in FEATURE_WEIGHTS that are not in binary_cols.
    for col in FEATURE_WEIGHTS:
        if col in df.columns and col not in binary_cols:
            col_min = df[col].min()
            col_max = df[col].max()
            # Avoid division by zero; if constant, set normalized value to 0.5.
            if col_max - col_min != 0:
                df[f"{col}_norm"] = (df[col] - col_min) / (col_max - col_min)
            else:
                df[f"{col}_norm"] = 0.5

    # --- Compute House Score Using Normalized Values ---
    # Multiply each normalized column by its contribution weight and sum them.
    house_score = sum(
        df[f"{col}_norm"] * FEATURE_WEIGHTS[col]
        for col in FEATURE_WEIGHTS if f"{col}_norm" in df.columns
    )
    df["house_score"] = house_score

    # If furnishing_contribution exists, add it to the house score
    if "furnishing_contribution" in df.columns:
        df["house_score"] += df["furnishing_contribution"]

    # --- Generate Targets ---
    # Generate a house type target by binning the house score into num_house_types classes.
    house_bins = np.linspace(df["house_score"].min(), df["house_score"].max(), num=num_house_types + 1)
    df["house_type"] = np.digitize(df["house_score"], bins=house_bins[:-1], right=True) - 1

    # Use the house score to derive the price category via quantile thresholds.
    quantileSet = [i / num_price_classes for i in range(1, num_price_classes)]
    house_score_bins = df["house_score"].quantile(quantileSet).values
    df["price_category"] = np.digitize(df["house_score"], bins=house_score_bins)

    print("New data house_score min:", df["house_score"].min())
    print("New data house_score max:", df["house_score"].max())
    print("Saved bins:", house_score_bins)

    # --- Prepare Training Features ---
    # Drop the raw "price", the target columns, and any engineered columns related to price.
    # We keep the computed house_score because it's a key feature.
    drop_cols = ["price", "price_category", "house_type"]
    if "furnishing_contribution" in df.columns:
        drop_cols.append("furnishing_contribution")
    # Also drop the normalized columns as they were used to compute house_score.
    norm_cols = [f"{col}_norm" for col in FEATURE_WEIGHTS if f"{col}_norm" in df.columns]
    drop_cols.extend(norm_cols)
    X = df.drop(columns=drop_cols)

    # Save the feature column names for later use
    feature_columns = X.columns.tolist()
    joblib.dump(feature_columns, "feature_columns.pkl")

    # Targets for training
    y_price = df["price_category"]
    y_type = df["house_type"]

    # Normalize the training features using MinMaxScaler (scale features to 0-1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)

    # Save the scaler for consistent transformation during prediction
    joblib.dump(scaler, "scaler.pkl")

    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train_price, y_test_price, y_train_type, y_test_type = train_test_split(
        X_scaled, y_price, y_type, test_size=0.2, random_state=42, stratify=y_price
    )

    output_file = "preprocessed_data.pkl"
    joblib.dump((X_train, X_test, y_train_price, y_test_price, y_train_type, y_test_type), output_file)

    return output_file

if __name__ == "__main__":
    output = preprocess_data()
    print("Preprocessed data saved to:", output)