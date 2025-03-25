import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

def preprocess_data(num_price_classes=5, num_house_types=5, fileName="Housing_Augmented.csv"):
    # Load dataset
    df = pd.read_csv(fileName)

    # Load environment variables
    load_dotenv()

    # Define contribution weights (used to compute the house score)
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

    # Convert binary categorical columns ("yes"/"no") to 1/0
    binary_cols = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
    df[binary_cols] = df[binary_cols].applymap(lambda x: 1 if str(x).strip().lower() == "yes" else 0)

    # One-hot encode 'furnishingstatus'
    df = pd.get_dummies(df, columns=["furnishingstatus"], drop_first=True)

    # Convert specified columns to numeric
    numeric_cols = ["price", "area", "bedrooms", "bathrooms", "stories", "parking"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()

    # Compute a house score using a weighted sum of features.
    # (The raw price is used here only for weighting but not later as a feature.)
    df["house_score"] = sum(
        df[col].astype(float) * FEATURE_WEIGHTS[col]
        for col in FEATURE_WEIGHTS if col in df.columns
    )

    # Generate a house type target by binning the house score into num_house_types classes.
    house_bins = np.linspace(df["house_score"].min(), df["house_score"].max(), num=num_house_types + 1)
    df["house_type"] = np.digitize(df["house_score"], bins=house_bins[:-1], right=True) - 1

    # Instead of using the raw price to generate a price category target,
    # use the house score to derive the price category.
    # Compute quantile thresholds for the house_score.
    quantileSet = [i / num_price_classes for i in range(1, num_price_classes)]
    house_score_bins = df["house_score"].quantile(quantileSet).values
    df["price_category"] = np.digitize(df["house_score"], bins=house_score_bins)

    # --- Prepare training features ---
    # Drop the raw "price", the target columns, and any engineered columns (if any) related to price.
    drop_cols = ["price", "price_category", "house_type"]
    # (If you had any engineered columns starting with "price_category_by_", drop them as well.)
    drop_cols += [col for col in df.columns if col.startswith("price_category_by_")]
    # After preparing X (the DataFrame used for training features):
    X = df.drop(columns=drop_cols)

    # Save the feature column names for later use
    feature_columns = X.columns.tolist()
    joblib.dump(feature_columns, "feature_columns.pkl")

    # Targets for training
    y_price = df["price_category"]
    y_type = df["house_type"]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save the scaler for consistent transformation during prediction
    joblib.dump(scaler, "scaler.pkl")

    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train_price, y_test_price, y_train_type, y_test_type = train_test_split(
        X_scaled, y_price, y_type, test_size=0.2, random_state=42, stratify=y_price
    )

    # Save preprocessed data to disk
    output_file = "preprocessed_data.pkl"
    joblib.dump((X_train, X_test, y_train_price, y_test_price, y_train_type, y_test_type), output_file)

    return output_file

if __name__ == "__main__":
    output = preprocess_data()
    print("Preprocessed data saved to:", output)